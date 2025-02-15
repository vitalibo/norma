from typing import Any, Callable, Dict, Iterable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import NumericType

from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
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


class MaskRule(Rule):
    """
    A rule to validate a column in a DataFrame.
    """

    def __init__(self, condition_func: Callable, error_type: str, error_msg: str):
        super().__init__()
        self.condition_func = condition_func
        self.error_type = error_type
        self.error_msg = error_msg

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        return df.withColumn(
            *error_state.add_errors(
                self.condition_func(column), column, {'type': self.error_type, 'msg': self.error_msg}
            )
        )


def required() -> Rule:
    """
    Checks if the input is missing.
    """

    class _RequiredRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            details = {'type': 'missing', 'msg': 'Field required'}

            if column not in df.columns:
                return df \
                    .withColumn(column, fn.lit(None).cast('string')) \
                    .withColumn(*error_state.add_errors(fn.lit(True), column, details=details))

            return df \
                .withColumn(*error_state.add_errors(fn.col(column).isNull(), column, details=details))

    return _RequiredRule()


def equal_to(eq: Any) -> Rule:
    """
    Checks if the input is equal to a given value.
    """

    return MaskRule(
        lambda col: fn.col(col) != fn.lit(eq),
        error_type='equal_to',
        error_msg=f'Input should be equal to {eq}'
    )


def not_equal_to(ne: Any) -> Rule:
    """
    Checks if the input is not equal to a given value.
    """

    return MaskRule(
        lambda col: fn.col(col) == fn.lit(ne),
        error_type='not_equal_to',
        error_msg=f'Input should not be equal to {ne}'
    )


def greater_than(gt: Any) -> Rule:
    """
    Checks if the input is greater than a given value.
    """

    return MaskRule(
        lambda col: fn.col(col) <= fn.lit(gt),
        error_type='greater_than',
        error_msg=f'Input should be greater than {gt}'
    )


def greater_than_equal(ge: Any) -> Rule:
    """
    Checks if the input is greater than or equal to a given value.
    """

    return MaskRule(
        lambda col: fn.col(col) < fn.lit(ge),
        error_type='greater_than_equal',
        error_msg=f'Input should be greater than or equal to {ge}'
    )


def less_than(lt: Any) -> Rule:
    """
    Checks if the input is less than a given value.
    """

    return MaskRule(
        lambda col: fn.col(col) >= fn.lit(lt),
        error_type='less_than',
        error_msg=f'Input should be less than {lt}'
    )


def less_than_equal(le: Any) -> Rule:
    """
    Checks if the input is less than or equal to a given value.
    """

    return MaskRule(
        lambda col: fn.col(col) > fn.lit(le),
        error_type='less_than_equal',
        error_msg=f'Input should be less than or equal to {le}'
    )


def multiple_of(multiple: Any) -> Rule:
    if not isinstance(multiple, (int, float)):
        raise ValueError('value should be an integer or a float')

    class _MultipleOfRule(MaskRule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if not isinstance(df.schema[column].dataType, NumericType):
                raise ValueError('multiple_of rule can only be applied to numeric columns')

            return super().verify(df, column, error_state)

    return _MultipleOfRule(
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        lambda col: (fn.col(col) % fn.lit(multiple)) != 0,
        error_type='multiple_of',
        error_msg=f'Input should be a multiple of {multiple}'
    )


def int_parsing() -> Rule:
    class _IntParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            return (
                df
                .withColumn(f'{column}_bak', fn.col(column))
                .withColumn(column, fn.col(column).cast('int'))
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(),
                        column,
                        details={
                            'type': 'int_parsing',
                            'msg': 'Input should be a valid integer, unable to parse value as an integer'
                        }
                    )
                )
            )

    return _IntParsingRule()


def float_parsing():
    class _FloatParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            return (
                df
                .withColumn(f'{column}_bak', fn.col(column))
                .withColumn(column, fn.col(column).cast('float'))
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(),
                        column,
                        details={
                            'type': 'float_parsing',
                            'msg': 'Input should be a valid float, unable to parse value as a float'
                        }
                    )
                )
            )

    return _FloatParsingRule()


def str_parsing() -> Rule:
    class _StrParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            return df \
                .withColumn(f'{column}_bak', fn.col(column)) \
                .withColumn(column, fn.col(column).cast('string'))

    return _StrParsingRule()


def bool_parsing() -> Rule:
    class _BoolParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            return (
                df
                .withColumn(f'{column}_bak', fn.col(column))
                .withColumn(column, fn.col(column).cast('boolean'))
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(),
                        column,
                        details={
                            'type': 'boolean_parsing',
                            'msg': 'Input should be a valid boolean, unable to parse value as a boolean'
                        }
                    )
                )
            )

    return _BoolParsingRule()


def date_parsing() -> Rule:
    raise NotImplementedError()


def datetime_parsing() -> Rule:
    raise NotImplementedError()


def min_length(value: int) -> Rule:
    """
    Checks if the input has a minimum length.
    """

    return MaskRule(
        lambda col: fn.length(fn.col(col)) < value,
        error_type='min_length',
        error_msg=f'Input should have a minimum length of {value}'
    )


def max_length(value: int) -> Rule:
    """
    Checks if the input has a maximum length.
    """

    return MaskRule(
        lambda col: fn.length(fn.col(col)) > value,
        error_type='max_length',
        error_msg=f'Input should have a maximum length of {value}'
    )


def pattern(regex: str) -> Rule:
    return MaskRule(
        lambda col: ~fn.col(col).rlike(regex),
        error_type='pattern',
        error_msg=f'Input should match the pattern {regex}'
    )


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    """
    A rule that forbids extra columns in the DataFrame.
    """

    class _ExtraRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if column in allowed:
                return df.withColumn(*error_state.add_errors(fn.lit(False), column, details=details))

            return (
                df
                .withColumnRenamed(column, f'{column}_bak')
                .withColumn(*error_state.add_errors(fn.lit(True), column, details=details))
            )

    details = {'type': 'extra_forbidden', 'msg': 'Extra inputs are not permitted'}
    return _ExtraRule()


def isin(values: Iterable[Any]) -> Rule:
    """
    Checks if the input is in a given list of values.
    """

    return MaskRule(
        lambda col: ~fn.col(col).isin(values),
        error_type='isin',
        error_msg=f'Input should be one of {values}'
    )


def notin(values: Iterable[Any]) -> Rule:
    """
    Checks if the input is not in a given list of values.
    """

    return MaskRule(
        lambda col: fn.col(col).isin(values),
        error_type='notin',
        error_msg=f'Input should not be one of {values}'
    )
