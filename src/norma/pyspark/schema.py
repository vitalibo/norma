from typing import Dict, List, Union

import pyspark.sql.functions as fn
from pyspark.sql import DataFrame

import norma.pyspark.rules
from norma.pyspark.rules import ErrorState, Rule


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
        dtype_rules = {
            'int': norma.pyspark.rules.int_parsing,
            'integer': norma.pyspark.rules.int_parsing,
            'float': norma.pyspark.rules.float_parsing,
            'double': norma.pyspark.rules.float_parsing,
            'number': norma.pyspark.rules.float_parsing,
            'string': norma.pyspark.rules.string_parsing,
            'str': norma.pyspark.rules.string_parsing,
            'boolean': norma.pyspark.rules.boolean_parsing,
            'bool': norma.pyspark.rules.boolean_parsing,
            # 'date': norma.pyspark.rules.date_parsing,
            # 'datetime': norma.pyspark.rules.datetime_parsing,
            # 'timestamp': norma.pyspark.rules.timestamp_parsing,
            # 'timestamp[s]': lambda: norma.pyspark.rules.timestamp_parsing('s'),
            # 'timestamp[ms]': lambda: norma.pyspark.rules.timestamp_parsing('ms'),
            # 'time': norma.pyspark.rules.time_parsing,
        }

        dtype = dtype.__name__ if isinstance(dtype, type) else dtype
        if dtype not in dtype_rules:
            raise ValueError(f"Unsupported dtype '{dtype}'")

        self.rules = [rules] if isinstance(rules, Rule) else rules
        self.rules.insert(0, dtype_rules[dtype]())


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
        error_state = ErrorState(error_column)
        for column in self.columns:
            df = df \
                .withColumn(f'{error_column}_{column}', fn.array())

            rules = self.columns[column].rules if column in self.columns else []
            if not self.allow_extra:
                rules.append(
                    norma.pyspark.rules.extra_forbidden(self.columns.keys()))

            for rule in rules:
                df = rule.verify(df, column, error_state)

        df = df.withColumn(error_column, fn.struct(*[
            fn.struct(
                fn.filter(fn.col(f'{error_column}_{column}'), fn.isnotnull).alias('details'),
                fn.col(column if column in df.columns else f'{column}_bak').alias('original'),
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
