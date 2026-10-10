from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as fn

import norma.rules
from norma.engines.pyspark.rules import ErrorState, default, extra_forbidden
from norma.engines.pyspark.utils import dtype_at, format_value, materialize, nested_get_expr, nested_set_expr

if TYPE_CHECKING:
    from norma.schema import Schema


def validate(schema: Schema, df: DataFrame, error_column: str) -> DataFrame:
    """
    Validate the PySpark DataFrame according to the schema

    :param schema: The schema to validate the DataFrame against
    :param df: The DataFrame to validate
    :param error_column: The name of the column to store error information
    """

    error_state = ErrorState(error_column, with_loc=schema.contains_arrays)
    error_state.resync(df)
    original_cols = df.columns

    df = _validate(df, schema, error_state, original_cols)
    df = error_state.flush(df)

    df = _format_error_details(df, error_state, error_column)
    df = _nullify_invalid_values(df, schema, error_state)

    kept_cols = [name for name in schema.columns if name in df.columns]
    if schema.allow_extra:
        kept_cols = [*original_cols, *(col for col in kept_cols if col not in original_cols)]
    return df.select(*kept_cols, error_column)


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

    columns = {
        name for name, column in schema.columns.items()
        if name in original_cols or column.required or column.has_default
    }
    if not schema.allow_extra:
        columns.update(original_cols)

    for column in sorted(columns):
        full_column = f'{parent}{column}'
        error_state.stub_if_absent(full_column)

        rules = []
        if column in schema.columns:
            if schema.columns[column].has_default:
                rules.append(default(schema.columns[column].default))
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

            inner_schema = inner_schema.inner_schema

        dtype = error_state.dtype_of(full_column)
        if dtype.typeName() != 'struct':
            continue

        df = _validate(df, inner_schema, error_state, dtype.fieldNames(), f'{full_column}.')

        if not error_state.dtype_of(full_column).fields:
            error_state.drop_column(full_column.removesuffix('[]'))

    return df


def _format_error_details(df, error_state, error_column) -> DataFrame:
    """
    Format the error details in the DataFrame
    """

    errors = {
        name: error_state.details_column_for(name)
        for name in list(error_state.names)
        if error_state.details_column_for(name) in df.columns
    }

    empty = fn.array()
    entries = [
        fn.when(
            fn.array_size(fn.col(error)) > 0,
            fn.array(fn.struct(
                fn.lit(name).alias('key'),
                fn.struct(
                    fn.col(error).alias('details'),
                    _make_origin(df, name, error_state).alias('original'),
                ).alias('value'),
            )),
        ).otherwise(empty)
        for name, error in errors.items()
    ]

    details_map = fn.map_from_entries(fn.concat(*entries)) if entries else fn.create_map()
    details_map = details_map.cast(error_state.errors_dtype())

    if error_state.extras:
        details_map = fn.map_concat(details_map, *error_state.extras.values())

    return df.withColumn(error_column, details_map)


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
        root = root_of(path)
        return nested_get_expr(path, exprs.get(root), dtypes.get(root))

    def set_expr(path, val):
        root = root_of(path)
        exprs[root] = nested_set_expr(path, val, exprs.get(root), dtypes.get(root))

    def apply(df):
        if not exprs:
            return df

        return df.select(*(
            materialize(exprs[column], dtypes[column]).alias(column) if column in exprs else fn.col(column)
            for column in df.columns
        ))

    return expr_of, set_expr, apply


def _nullify_invalid_values(df, schema, error_state):
    """
    Reset invalid values to null after validation
    """

    expr_of, set_expr, apply = _expr_accumulator(df)

    def reset_elements(column, flags):
        try:
            dtype = dtype_at(df.schema, column)
        except KeyError:
            # the column was dropped along the way, while the error flags of its elements remained
            return
        set_expr(column, _nullify_elements(expr_of(column), flags, dtype.elementType))

    masks = {}
    for name, _ in reversed(schema.nested_columns.items()):
        if name in masks:
            # the array column comes right after its element columns: its elements are reset first, so that the checks
            # of the array itself and of its parents wrap them instead of being inlined into every field around them
            reset_elements(name, masks.pop(name))
        if name not in error_state.names:
            continue
        if '[]' not in name and error_state.details_column_for(name) in df.columns:
            set_expr(name, fn.when(fn.array_size(fn.col(error_state.details_column_for(name))) <= 0, expr_of(name)))
        elif '[]' in name and error_state.indexes_column_for(name) in df.columns:
            root, *nested = name.split('[].')
            path = nested[0] if nested else ''
            masks.setdefault(root.removesuffix('[]'), []).append((path, fn.col(error_state.indexes_column_for(name))))

    return apply(df)


def _nullify_elements(array, flags, element_dtype):
    """
    Reset the invalid values held by the elements of an array
    """

    zipped = fn.arrays_zip(array.alias('_0'), *(flag.alias(f'_{i}') for i, (_, flag) in enumerate(flags, start=1)))

    def element_of(item):
        value = item['_0']
        for i, (path, _) in enumerate(flags, start=1):
            flag = item[f'_{i}']
            if not path:
                value = fn.when(~flag, materialize(value, element_dtype))
                continue
            value = nested_set_expr(f'_.{path}', lambda x, flag=flag: fn.when(~flag, x), value, element_dtype)
        return materialize(value, element_dtype)

    return fn.transform(zipped, element_of)


def _make_origin(df, column, error_state):
    """
    Format the original value of a column for error reporting
    """

    backup = error_state.backup_column_for(column)
    if backup in df.columns:
        column = backup

    dtype = dtype_at(df.schema, column).typeName()

    if '[]' in column:
        return format_value(nested_get_expr(column), 'array')
    return format_value(fn.col(column), dtype)
