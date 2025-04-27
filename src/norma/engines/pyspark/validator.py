from pyspark.sql import DataFrame
from pyspark.sql import functions as fn

import norma.rules
from norma.engines.pyspark.rules import ErrorState, data_type_of, extra_forbidden
from norma.engines.pyspark.utils import backup_col, suffix_col, with_nested_column


def validate(
        schema: 'Schema', df: DataFrame, error_column: str
) -> DataFrame:
    """
    Validate the PySpark DataFrame according to the schema

    :param schema: The schema to validate the DataFrame against
    :param df: The DataFrame to validate
    :param error_column: The name of the column to store error information
    """

    error_state = ErrorState(error_column)
    original_cols = df.columns
    df = _validate_df(df, schema, error_state, original_cols)

    df = df.withColumn(error_column, fn.struct(*[
        fn.struct(
            fn.filter(fn.col(f'{error_column}_{suffix}'), fn.isnotnull).alias('details'),
            _make_origin(df, column, error_state).alias('original'),
        ).alias(suffix) for column, suffix in error_state.suffixes.items()
    ]))

    for name, column in reversed(schema.nested_columns.items()):
        df = df.transform(
            with_nested_column(name, fn.when(
                fn.array_size(fn.col(f'{error_column}.{suffix_col(name, error_state)}.details')) > 0, None
            ).otherwise(fn.col(name)))
        )

    df = df.select(
        *(original_cols if schema.allow_extra else schema.columns),
        fn.map_filter(
            fn.map_from_arrays(
                fn.array(*[fn.lit(column) for column, suffix in error_state.suffixes.items()]),
                fn.array(*[
                    fn.when(
                        fn.array_size(fn.col(f'{error_column}.{suffix}.details')) > 0,
                        fn.col(f'{error_column}.{suffix}')
                    ).otherwise(fn.lit(None)).alias(suffix)
                    for column, suffix in error_state.suffixes.items()
                ])
            ).alias(error_column),
            lambda k, v: fn.isnotnull(v)
        ).alias(error_column)
    )

    df = df.fillna({name: col.default for name, col in schema.columns.items() if col.default is not None})

    for name, col in schema.nested_columns.items():
        if col.default is not None:
            df = df.transform(with_nested_column(name, fn.coalesce(fn.col(name), fn.lit(col.default))))

    for name, col in schema.nested_columns.items():
        if col.default_factory is not None:
            df = df.transform(with_nested_column(name, fn.coalesce(fn.col(name), col.default_factory(df))))

    if not schema.allow_extra:
        return df.select(*list(schema.columns.keys()), error_column)
    return df.select(*original_cols, error_column)


def _validate_df(df, schema, error_state, original_cols, parent=''):
    """
    Go through the schema and validate each column
    """

    columns_with_extra = set(schema.columns.keys())
    if not schema.allow_extra:
        columns_with_extra = set(original_cols + list(schema.columns.keys()))

    for column in columns_with_extra:
        suffix = suffix_col(f'{parent}{column}', error_state)
        df = df.withColumn(f'{error_state.error_column}_{suffix}', fn.array())

        rules = schema.columns[column].rules if column in schema.columns else []
        if not schema.allow_extra:
            rules.append(
                extra_forbidden([f'{parent}{o}' for o in schema.columns.keys()]))

        for rule in rules:
            if isinstance(rule, norma.rules.RuleProxy):
                rule = getattr(norma.engines.pyspark.rules, rule.name)(**rule.kwargs)

            df = rule.verify(df, f'{parent}{column}', error_state)

        if column not in schema.columns or schema.columns[column].inner_schema is None:
            continue

        df = _validate_df(
            df,
            schema.columns[column].inner_schema,
            error_state,
            data_type_of(df, f'{parent}{column}').fieldNames(),
            parent=f'{parent}{column}.'
        )

    return df


def _make_origin(df: DataFrame, column, error_state):
    """
    Format the original value of a column for error reporting
    """

    backup_column = backup_col(column, error_state)
    column = backup_column if backup_column in df.columns else column
    dtype = data_type_of(df, column).typeName()

    null = fn.when(fn.col(column).isNull(), fn.lit('null'))
    if dtype in ('string',):
        return null.otherwise(fn.concat(fn.lit('"'), fn.col(column), fn.lit('"')))
    elif dtype in ('array', 'map', 'struct'):
        return null.otherwise(fn.to_json(fn.col(column)))

    return null.otherwise(fn.col(column).cast('string'))
