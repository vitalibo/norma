from pyspark.sql import DataFrame
from pyspark.sql import functions as fn

import norma.rules
from norma.engines.pyspark.rules import ErrorState, extra_forbidden


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
    columns_with_extra = \
        set(schema.columns.keys() if schema.allow_extra else original_cols + list(schema.columns.keys()))

    for column in columns_with_extra:
        df = df \
            .withColumn(f'{error_column}_{column}', fn.array())

        rules = schema.columns[column].rules if column in schema.columns else []
        if not schema.allow_extra:
            rules.append(
                extra_forbidden(schema.columns.keys()))

        for rule in rules:
            if isinstance(rule, norma.rules.RuleProxy):
                rule = getattr(norma.engines.pyspark.rules, rule.name)(**rule.kwargs)

            df = rule.verify(df, column, error_state)

    df = df.withColumn(error_column, fn.struct(*[
        fn.struct(
            fn.filter(fn.col(f'{error_column}_{column}'), fn.isnotnull).alias('details'),
            _make_origin(df, column).alias('original'),
        ).alias(column) for column in columns_with_extra
    ]))

    df = df.select(
        *(set(original_cols) - set(schema.columns.keys()) if schema.allow_extra else []),
        *[
            fn.when(
                fn.array_size(fn.col(f'{error_column}.{column}.details')) > 0, None
            ).otherwise(fn.col(column)).alias(column)
            for column in schema.columns
        ],
        fn.map_filter(
            fn.map_from_arrays(
                fn.array(*[fn.lit(column) for column in columns_with_extra]),
                fn.array(*[
                    fn.when(
                        fn.array_size(fn.col(f'{error_column}.{column}.details')) > 0,
                        fn.col(f'{error_column}.{column}')
                    ).otherwise(fn.lit(None)).alias(column)
                    for column in columns_with_extra
                ])
            ).alias(error_column),
            lambda k, v: fn.isnotnull(v)
        ).alias(error_column)
    )

    df = df.fillna({name: col.default for name, col in schema.columns.items() if col.default is not None})

    for name, col in schema.columns.items():
        if col.default_factory is not None:
            df = df.withColumn(name, fn.coalesce(fn.col(name), col.default_factory(df)))

    if not schema.allow_extra:
        return df.select(*list(schema.columns.keys()), error_column)
    return df.select(*original_cols, error_column)


def _make_origin(df: DataFrame, column):
    """
    Format the original value of a column for error reporting
    """

    column = f'{column}_bak' if f'{column}_bak' in df.columns else column
    dtype = df.schema[column].dataType.typeName()

    null = fn.when(fn.col(column).isNull(), fn.lit('null'))
    if dtype in ('string',):
        return null.otherwise(fn.concat(fn.lit('"'), fn.col(column), fn.lit('"')))
    elif dtype in ('array', 'map', 'struct'):
        return null.otherwise(fn.to_json(fn.col(column)))

    return null.otherwise(fn.col(column).cast('string'))
