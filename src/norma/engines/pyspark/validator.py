from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as fn

import norma.rules
from norma.engines.pyspark.rules import ErrorState, extra_forbidden
from norma.engines.pyspark.utils import dtype_at, nested_get_expr, nested_set_expr

if TYPE_CHECKING:
    from norma.schema import Schema


def validate(schema: Schema, df: DataFrame, error_column: str) -> DataFrame:
    """
    Validate the PySpark DataFrame according to the schema

    :param schema: The schema to validate the DataFrame against
    :param df: The DataFrame to validate
    :param error_column: The name of the column to store error information
    """

    error_state = ErrorState(error_column, schema)
    error_state.seed()
    error_state.resync(df)
    original_cols = df.columns

    df = _validate(df, schema, error_state, original_cols)
    df = error_state.flush(df)

    df = _format_error_details(df, error_state)
    df = _nullify_invalid_values(df, schema, error_state)
    df = _fill_defaults(df, schema)

    return df.select(*(original_cols if schema.allow_extra else schema.columns), error_column)


def _apply_rule(df, rule, column, error_state):
    """
    Apply a single rule to the DataFrame.
    """

    if isinstance(rule, norma.rules.RuleProxy):
        rule = getattr(norma.engines.pyspark.rules, rule.name)(**rule.kwargs)

    if getattr(rule, 'accumulates', False):
        return rule.verify(df, column, error_state)

    df = error_state.flush(df)
    df = rule.verify(df, column, error_state)
    error_state.resync(df)
    return df


def _validate(df, schema, error_state, original_cols, parent=''):
    """
    Recursively validate the DataFrame according to the schema
    """

    columns = set(schema.columns.keys())
    if not schema.allow_extra:
        columns.update(original_cols)

    for column in columns:
        full_column = f'{parent}{column}'
        error_state.track_column(full_column)

        rules = []
        if column in schema.columns:
            rules.extend(schema.columns[column].rules)
        if not schema.allow_extra:
            rules.append(extra_forbidden([f'{parent}{o}' for o in schema.columns]))

        for rule in rules:
            df = _apply_rule(df, rule, full_column, error_state)

        # if there is inner schema, validate recursively
        if column not in schema.columns or schema.columns[column].inner_schema is None:
            continue
        inner_schema = schema.columns[column].inner_schema

        if schema.columns[column].dtype in {'array', 'list'}:
            full_column = f'{full_column}[]'

            for rule in inner_schema.rules:
                df = _apply_rule(df, rule, full_column, error_state)

            dtype = error_state.dtype_of(full_column)
            if dtype.typeName() != 'struct':
                continue

            df = _validate(df, inner_schema.inner_schema, error_state, dtype.fieldNames(), f'{full_column}.')
        else:
            dtype = error_state.dtype_of(full_column)
            df = _validate(df, inner_schema, error_state, dtype.fieldNames(), f'{full_column}.')

    return df


def _format_error_details(df, error_state) -> DataFrame:
    """
    Format the error details in the DataFrame
    """

    error_column = error_state.error_column
    errors = {
        name: error_state.errors_col(name)
        for name in list(error_state.names)
        if error_state.errors_col(name) in df.columns
    }

    return df \
        .withColumns({details: fn.filter(fn.col(details), fn.isnotnull) for details in errors.values()}) \
        .withColumn(
            error_column, fn.map_filter(
                fn.map_from_arrays(
                    fn.array(*[fn.lit(name) for name in errors]),
                    fn.array(*[
                        fn.when(
                            fn.array_size(fn.col(error)) > 0,
                            fn.struct(
                                fn.col(error).alias('details'),
                                _make_origin(df, name, error_state).alias('original'),
                            ),
                        )
                        for name, error in errors.items()
                    ]),
                ),
                lambda _, v: fn.isnotnull(v),
            ),
        )


def _expr_accumulator(df):
    """
    Create a local expression accumulator that composes nested column updates per root column,
    materializing them with a single projection.
    """

    dtypes = {field.name: field.dataType for field in df.schema.fields}
    exprs = {}

    def root_of(path):
        root = path.split('.')[0]
        return root.removesuffix('[]')

    def expr_of(path):
        return nested_get_expr(path, exprs.get(root_of(path)))

    def set_expr(path, val):
        root = root_of(path)
        exprs[root] = nested_set_expr(path, val, exprs.get(root), dtypes.get(root))

    def apply(df):
        if not exprs:
            return df
        return df.select(*(exprs.get(column, fn.col(column)).alias(column) for column in df.columns))

    return expr_of, set_expr, apply


def _nullify_invalid_values(df, schema, error_state):
    """
    Reset invalid values to null after validation
    """

    expr_of, set_expr, apply = _expr_accumulator(df)

    def nested_zip(x, y, nodes):
        if not nodes:
            return fn.when(~y, x)

        return x.withField(nodes[0], nested_zip(x.getField(nodes[0]), y, nodes[1:]))

    for name, _ in reversed(schema.nested_columns.items()):
        if '[]' not in name:
            set_expr(name, fn.when(fn.array_size(fn.col(error_state.errors_col(name))) <= 0, expr_of(name)))
        elif error_state.indexes_col(name) in df.columns:
            root, *nested = name.split('[].')
            root = root.removesuffix('[]')
            nested = nested[0].split('.') if nested else []
            set_expr(
                root, fn.zip_with(
                    expr_of(root), fn.col(error_state.indexes_col(name)), lambda x, y: nested_zip(x, y, nested)  # noqa: B023
                ),
            )

    return apply(df)


def _fill_defaults(df, schema):
    """
    Fill null with default values in the DataFrame according to the schema
    """

    def default_as_lit(column):
        if column.dtype == 'date':
            return fn.lit(column.default).cast('date')
        if column.dtype == 'datetime':
            return fn.lit(column.default).cast('timestamp')
        return fn.lit(column.default)

    def default_if_null(default):
        def wrap(column):
            return fn.coalesce(column, default)

        return wrap

    expr_of, set_expr, apply = _expr_accumulator(df)
    for name, col in schema.nested_columns.items():
        if col.default is None:
            continue
        if '[]' not in name:
            set_expr(name, fn.coalesce(expr_of(name), default_as_lit(col)))
        else:
            set_expr(name, default_if_null(default_as_lit(col)))
    df = apply(df)

    for name, col in schema.nested_columns.items():
        if col.default_factory is None:
            continue
        root = name.split('.')[0].removesuffix('[]')
        dtype = df.schema[root].dataType if root in df.columns else None
        if '[]' not in name:
            val = fn.coalesce(fn.col(name), col.default_factory(df))
        else:
            val = default_if_null(col.default_factory(df))
        df = df.withColumn(root, nested_set_expr(name, val, None, dtype))

    return df


def _make_origin(df, column, error_state):
    """
    Format the original value of a column for error reporting
    """

    def format_value(value, dtype_name):
        null = fn.when(value.isNull(), fn.lit('null'))
        if dtype_name == 'string':
            return null.otherwise(fn.concat(fn.lit('"'), value, fn.lit('"')))
        if dtype_name in {'array', 'map', 'struct'}:
            return null.otherwise(fn.to_json(value))
        return null.otherwise(value.cast('string'))

    backup_column = error_state.backup_col(column)
    if backup_column in df.columns:
        column = backup_column

    dtype = dtype_at(df.schema, column).typeName()

    if '[]' in column:
        return format_value(nested_get_expr(column), 'array')
    return format_value(fn.col(column), dtype)
