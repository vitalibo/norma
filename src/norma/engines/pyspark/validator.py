from pyspark.sql import DataFrame
from pyspark.sql import functions as fn

import norma.rules
from norma.engines.pyspark.rules import ErrorState, data_type_of, extra_forbidden
from norma.engines.pyspark.utils import (
    backup_col, flatten_nested_values, suffix_col, with_nested_column, zip_with_nested_columns
)


def validate(
        schema: 'Schema', df: DataFrame, error_column: str
) -> DataFrame:
    """
    Validate the PySpark DataFrame according to the schema

    :param schema: The schema to validate the DataFrame against
    :param df: The DataFrame to validate
    :param error_column: The name of the column to store error information
    """

    error_state = ErrorState(error_column, schema)
    original_cols = df.columns

    df = _validate(df, schema, error_state, original_cols)
    df = _format_error_details(df, error_state)
    df = _nullify_invalid_values(df, schema, error_state)
    df = _fill_defaults(df, schema)

    return df.select(*(original_cols if schema.allow_extra else schema.columns), error_column)


def _validate(df, schema, error_state, original_cols, parent=''):
    """
    Recursively validate the DataFrame according to the schema
    """

    columns = set(schema.columns.keys())
    if not schema.allow_extra:
        columns.update(original_cols)

    for column in columns:  # pylint: disable=too-many-nested-blocks
        full_column = f'{parent}{column}'
        df = error_state.initialize_column(df, full_column)

        rules = []
        if column in schema.columns:
            rules.extend(schema.columns[column].rules)
        if not schema.allow_extra:
            rules.append(
                extra_forbidden([f'{parent}{o}' for o in schema.columns.keys()]))

        for rule in rules:
            if isinstance(rule, norma.rules.RuleProxy):
                rule = getattr(norma.engines.pyspark.rules, rule.name)(**rule.kwargs)
            df = rule.verify(df, full_column, error_state)

        # if there is inner schema, validate recursively
        if column not in schema.columns \
                or schema.columns[column].inner_schema is None:
            continue
        inner_schema = schema.columns[column].inner_schema

        if schema.columns[column].dtype in ('array', 'list'):
            full_column = f'{full_column}[]'

            for rule in inner_schema.rules:
                if isinstance(rule, norma.rules.RuleProxy):
                    rule = getattr(norma.engines.pyspark.rules, rule.name)(**rule.kwargs)
                df = rule.verify(df, full_column, error_state)

            dtype = data_type_of(df, full_column)
            if dtype.typeName() != 'struct':
                continue

            df = _validate(df, inner_schema.inner_schema, error_state, dtype.fieldNames(), f'{full_column}.')
        else:
            dtype = data_type_of(df, full_column)
            df = _validate(df, inner_schema, error_state, dtype.fieldNames(), f'{full_column}.')

    return df


def _format_error_details(df, error_state) -> DataFrame:
    """
    Format the error details in the DataFrame
    """

    error_column = error_state.error_column
    errors = {
        name: column for name, column in (
            (name, f'{error_column}_{suffix}') for name, suffix in error_state.suffixes.items()
        ) if column in df.columns
    }

    return df \
        .withColumns({details: fn.filter(fn.col(details), fn.isnotnull) for details in errors.values()}) \
        .withColumn(error_column, fn.map_filter(
        fn.map_from_arrays(
            fn.array(*[fn.lit(name) for name in errors]),
            fn.array(*[
                fn.when(
                    fn.array_size(fn.col(error)) > 0,
                    fn.struct(
                        fn.col(error).alias('details'),
                        _make_origin(df, name, error_state).alias('original'),
                    )
                )
                for name, error in errors.items()
            ])
        ),
        lambda k, v: fn.isnotnull(v)
    ))


def _nullify_invalid_values(df, schema, error_state):
    """
    Reset invalid values to null after validation
    """

    error_column = error_state.error_column
    for name, _ in reversed(schema.nested_columns.items()):
        suffix = suffix_col(name, error_state)
        if '[]' not in name:
            df = df.transform(with_nested_column(
                name, fn.when(fn.array_size(fn.col(f'{error_column}_{suffix}')) <= 0, fn.col(name))))
        elif f'{suffix}_indexes' in df.columns:
            df = df.transform(zip_with_nested_columns(
                name, fn.col(f'{suffix}_indexes'), lambda x, y: fn.when(~y, x)))
    return df


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

    for name, col in schema.nested_columns.items():
        if col.default is None:
            continue
        if '[]' not in name:
            df = df.transform(with_nested_column(name, fn.coalesce(fn.col(name), default_as_lit(col))))
        else:
            df = df.transform(with_nested_column(name, default_if_null(default_as_lit(col))))

    for name, col in schema.nested_columns.items():
        if col.default_factory is None:
            continue
        if '[]' not in name:
            df = df.transform(with_nested_column(name, fn.coalesce(fn.col(name), col.default_factory(df))))
        else:
            df = df.transform(with_nested_column(name, default_if_null(col.default_factory(df))))

    return df


def _make_origin(df, column, error_state):
    """
    Format the original value of a column for error reporting
    """

    def format_value(value, dtype_name):
        null = fn.when(value.isNull(), fn.lit('null'))
        if dtype_name in ('string',):
            return null.otherwise(fn.concat(fn.lit('"'), value, fn.lit('"')))
        elif dtype_name in ('array', 'map', 'struct'):
            return null.otherwise(fn.to_json(value))
        return null.otherwise(value.cast('string'))

    backup_column = backup_col(column, error_state)
    if backup_column in df.columns:
        column = backup_column

    dtype = data_type_of(df, column).typeName()

    if '[]' in column:
        return format_value(flatten_nested_values(column), 'array')
    return format_value(fn.col(column), dtype)
