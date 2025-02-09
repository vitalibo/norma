from typing import Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as fn

from norma.engines.pyspark.rules import ErrorState, extra_forbidden
from norma.rules import Rule


def validate(
        columns: Dict[str, List[Rule]], df: DataFrame, allow_extra: bool, error_column: str = 'errors'
) -> DataFrame:
    error_state = ErrorState(error_column)
    for column in columns:
        df = df \
            .withColumn(f'{error_column}_{column}', fn.array())

        rules = columns[column] if column in columns else []
        if not allow_extra:
            rules.append(
                extra_forbidden(columns.keys()))

        for rule in rules:
            df = rule.verify(df, column, error_state)

    df = df.withColumn(error_column, fn.struct(*[
        fn.struct(
            fn.filter(fn.col(f'{error_column}_{column}'), fn.isnotnull).alias('details'),
            fn.col(f'{column}_bak' if f'{column}_bak' in df.columns else column).alias('original'),
        ).alias(column) for column in columns
    ]))

    df = df.select(
        *[
            fn.when(
                fn.array_size(fn.col(f'{error_column}.{column}.details')) > 0, None
            ).otherwise(fn.col(column)).alias(column)
            for column in columns
        ],
        fn.map_filter(
            fn.map_from_arrays(
                fn.array(*[fn.lit(column) for column in columns]),
                fn.array(*[
                    fn.when(
                        fn.array_size(fn.col(f'{error_column}.{column}.details')) > 0,
                        fn.col(f'{error_column}.{column}')
                    ).otherwise(fn.lit(None)).alias(column)
                    for column in columns
                ])
            ).alias(error_column),
            lambda k, v: fn.isnotnull(v)
        ).alias(error_column)
    )

    return df
