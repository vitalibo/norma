from typing import Any, Callable, Dict

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn


class ErrorState:
    """
    A class to represent the state of errors in a DataFrame.
    """

    def __init__(self, error_column: str):
        self.error_column = error_column

    def add_errors(self, boolmask: Column, column: str, details: Dict[str, str]):
        details_col = fn.when(
            boolmask, fn.struct(*[
                fn.lit(v).alias(k)
                for k, v in details.items()
            ])
        )

        name = f'{self.error_column}_{column}'
        return name, fn.array_append(fn.col(name), details_col)


class Rule:
    """
    A rule to validate a column in a DataFrame.
    """

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        pass


class MaskRule(Rule):
    """
    A rule to validate a column in a DataFrame.
    """

    def __init__(self, condition_func: Callable, error_type: str, error_msg: str):
        self.condition_func = condition_func
        self.error_type = error_type
        self.error_msg = error_msg

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        return df.withColumn(
            *error_state.add_errors(
                self.condition_func(column), column, {'type': self.error_type, 'msg': self.error_msg}
            )
        )


def required():
    return MaskRule(
        lambda col: fn.col(col).isNull(),
        error_type='required',
        error_msg='Input is required'
    )


def equal_to(value: Any) -> Rule:
    return MaskRule(
        lambda col: fn.col(col) != fn.lit(value),
        error_type='equal_to',
        error_msg=f'Input should be equal to {value}'
    )


def eq(value: Any) -> Rule:
    return equal_to(value)


def not_equal_to(value: Any) -> Rule:
    return MaskRule(
        lambda col: fn.col(col) == fn.lit(value),
        error_type='not_equal_to',
        error_msg=f'Input should not be equal to {value}'
    )


def ne(value: Any) -> Rule:
    return not_equal_to(value)


def greater_than(value: Any) -> Rule:
    return MaskRule(
        lambda col: fn.col(col) <= fn.lit(value),
        error_type='greater_than',
        error_msg=f'Input should be greater than {value}'
    )


def multiple_of(value: Any) -> Rule:
    return MaskRule(
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        lambda col: (fn.col(col) % fn.lit(value)) != 0,
        error_type='multiple_of',
        error_msg=f'Input should be a multiple of {value}'
    )


def pattern(value: str) -> Rule:
    return MaskRule(
        lambda col: ~fn.col(col).rlike(value),
        error_type='pattern',
        error_msg=f'Input should match the pattern {value}'
    )


def int_parsing() -> Rule:
    class _IntParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            return (
                df
                .withColumn(column, fn.col(column).cast('int'))
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f"{column}_bak").isNotNull(),
                        column,
                        details={
                            'type': 'int_parsing',
                            'msg': 'Input should be a valid integer, unable to parse value as an integer'
                        }
                    )
                )
            )

    return _IntParsingRule()


def string_parsing() -> Rule:
    class _StrParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            return df \
                .withColumn(column, fn.col(column).cast('string'))

    return _StrParsingRule()
