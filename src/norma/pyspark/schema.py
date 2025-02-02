from typing import Dict, List, Union

import pyspark.sql.functions as fn
from pyspark.sql import DataFrame

from norma.pyspark.rules import Rule


class Column:
    """
    A column in a DataFrame schema.
    """

    def __init__(
            self,
            dtype: Union[type, str],
            *,
            rules: Union[Rule, List[Rule], None] = None
    ) -> None:
        self.dtype = dtype
        self.rules = [rules] if isinstance(rules, Rule) else rules


class Schema:
    """
    A schema for a DataFrame.
    """

    def __init__(
            self,
            columns: Dict[str, Column],
            allow_extra: bool = False
    ) -> None:
        self.columns = columns
        self.allow_extra = allow_extra

    def validate(self, df: DataFrame, error_column: str = 'errors') -> DataFrame:
        df = df.withColumn(error_column, fn.struct(*[
            fn.struct(
                fn.filter(
                    fn.array(*[
                        rule.expr(column)
                        for rule in self.columns[column].rules
                    ]),
                    fn.isnotnull,
                ).alias('details'),
                fn.col(column).alias('original'),
            ).alias(column) for column in self.columns
        ]))

        df = df.select(
            *[
                fn.when(
                    fn.array_size(fn.col(f'{error_column}.{column}.details')) > 0, None
                ).otherwise(fn.col(column)).alias(column)
                for column in self.columns
            ],
            fn.struct(*[
                fn.when(
                    fn.array_size(fn.col(f'{error_column}.{column}.details')) > 0,
                    fn.col(f'{error_column}.{column}')
                ).otherwise(fn.lit(None)).alias(column)
                for column in self.columns
            ]).alias(error_column)
        )

        return df
