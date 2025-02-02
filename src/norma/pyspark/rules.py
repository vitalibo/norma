from typing import Any, Callable

from pyspark.sql import Column
from pyspark.sql import functions as fn


class Rule:
    """
    A rule to validate a column in a DataFrame.
    """

    def __init__(self, condition_func: Callable, error_type: str, error_msg: str):
        self.condition_func = condition_func
        self.error_type = error_type
        self.error_msg = error_msg

    def expr(self, col: str):
        return fn.when(
            self.condition_func(col),
            fn.struct(
                fn.lit(self.error_type).alias('type'),
                fn.lit(self.error_msg).alias('msg')))

    def cast(self, col: Column):
        return col


def required():
    return Rule(
        lambda col: fn.col(col).isNull(),
        error_type='required',
        error_msg='Input is required'
    )


def equal_to(value: Any) -> Rule:
    return Rule(
        lambda col: fn.col(col) != fn.lit(value),
        error_type='equal_to',
        error_msg=f'Input should be equal to {value}'
    )


def eq(value: Any) -> Rule:
    return equal_to(value)


def not_equal_to(value: Any) -> Rule:
    return Rule(
        lambda col: fn.col(col) == fn.lit(value),
        error_type='not_equal_to',
        error_msg=f'Input should not be equal to {value}'
    )


def ne(value: Any) -> Rule:
    return not_equal_to(value)


def greater_than(value: Any) -> Rule:
    return Rule(
        lambda col: fn.col(col) <= fn.lit(value),
        error_type='greater_than',
        error_msg=f'Input should be greater than {value}'
    )


def multiple_of(value: Any) -> Rule:
    return Rule(
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        lambda col: (fn.col(col) % fn.lit(value)) != 0,
        error_type='multiple_of',
        error_msg=f'Input should be a multiple of {value}'
    )


def pattern(value: str) -> Rule:
    return Rule(
        lambda col: ~fn.col(col).rlike(value),
        error_type='pattern',
        error_msg=f'Input should match the pattern {value}'
    )


def int_parsing() -> Rule:
    class _IntParsingRule(Rule):
        def cast(self, col: Column):
            return col.cast('int')

    return _IntParsingRule(
        lambda col: fn.col(col).cast('int').isNull(),
        error_type='int_parsing',
        error_msg='Input should be a valid integer, unable to parse value as an integer'
    )


def string_parsing() -> Rule:
    class _StrRule(Rule):
        def cast(self, col: Column):
            return col.cast('string')

    return _StrRule(
        lambda col: fn.col(col).isNull(),
        error_type='string_parsing',
        error_msg='Input should be a valid string, unable to parse value as a string'
    )
