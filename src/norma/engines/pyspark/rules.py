import re
from typing import Any, Callable, Dict, Iterable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import BooleanType, DateType, FloatType, IntegerType, NumericType, StringType, TimestampType

from norma.errors import (
    BOOL_PARSING,
    BOOL_TYPE,
    DATE_PARSING,
    DATE_TYPE,
    DATETIME_PARSING,
    DATETIME_TYPE,
    ENUM,
    EQUAL_TO,
    EXTRA_FORBIDDEN,
    FLOAT_PARSING,
    FLOAT_TYPE,
    GREATER_THAN,
    GREATER_THAN_EQUAL,
    INT_PARSING,
    INT_TYPE,
    LESS_THAN,
    LESS_THAN_EQUAL,
    MISSING,
    MULTIPLE_OF,
    NOT_ENUM,
    NOT_EQUAL_TO,
    STRING_PATTERN_MISMATCH,
    STRING_TOO_LONG,
    STRING_TOO_SHORT,
    STRING_TYPE
)
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

    def __init__(self, condition_func: Callable, type: str, msg: str):  # noqa pylint: disable=redefined-builtin
        super().__init__()
        self.condition_func = condition_func
        self.type = type
        self.msg = msg

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        return df.withColumn(
            *error_state.add_errors(
                self.condition_func(column), column, {'type': self.type, 'msg': self.msg}
            )
        )


def required() -> Rule:
    """
    Checks if the input is missing.
    """

    class _RequiredRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if column not in df.columns:
                return df \
                    .withColumn(column, fn.lit(None).cast('string')) \
                    .withColumn(*error_state.add_errors(fn.lit(True), column, details=MISSING))

            return df \
                .withColumn(*error_state.add_errors(fn.col(column).isNull(), column, details=MISSING))

    return _RequiredRule()


def equal_to(eq: Any) -> Rule:
    """
    Checks if the input is equal to a given value.
    """

    if eq is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda col: fn.col(col) != fn.lit(eq),
        **EQUAL_TO.format(eq=eq)
    )


def not_equal_to(ne: Any) -> Rule:
    """
    Checks if the input is not equal to a given value.
    """

    if ne is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda col: fn.col(col) == fn.lit(ne),
        **NOT_EQUAL_TO.format(ne=ne)
    )


def greater_than(gt: Any) -> Rule:
    """
    Checks if the input is greater than a given value.
    """

    if gt is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda col: fn.col(col) <= fn.lit(gt),
        **GREATER_THAN.format(gt=gt)
    )


def greater_than_equal(ge: Any) -> Rule:
    """
    Checks if the input is greater than or equal to a given value.
    """

    if ge is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda col: fn.col(col) < fn.lit(ge),
        **GREATER_THAN_EQUAL.format(ge=ge)
    )


def less_than(lt: Any) -> Rule:
    """
    Checks if the input is less than a given value.
    """

    if lt is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda col: fn.col(col) >= fn.lit(lt),
        **LESS_THAN.format(lt=lt)
    )


def less_than_equal(le: Any) -> Rule:
    """
    Checks if the input is less than or equal to a given value.
    """

    if le is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda col: fn.col(col) > fn.lit(le),
        **LESS_THAN_EQUAL.format(le=le)
    )


def multiple_of(multiple: Any) -> Rule:
    if multiple is None:
        raise ValueError('multiple_of must not be None')
    if not isinstance(multiple, (int, float)):
        raise ValueError('multiple_of must be an integer or a float')

    class _MultipleOfRule(MaskRule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if not isinstance(df.schema[column].dataType, NumericType):
                raise ValueError('multiple_of rule can only be applied to numeric columns')

            return super().verify(df, column, error_state)

    return _MultipleOfRule(
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        lambda col: (fn.col(col) % fn.lit(multiple)) != 0,
        **MULTIPLE_OF.format(multiple_of=multiple)
    )


def int_parsing() -> Rule:
    class _IntParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if isinstance(df.schema[column].dataType, IntegerType):
                return df

            df = df.withColumn(f'{column}_bak', fn.col(column))

            if not isinstance(df.schema[column].dataType, (StringType, NumericType, BooleanType)):
                return df.withColumn(*error_state.add_errors(fn.lit(True), column, details=INT_TYPE))

            return (
                df.withColumn(column, fn.col(column).cast('int'))
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column, details=INT_PARSING
                    )
                )
            )

    return _IntParsingRule()


def float_parsing():
    class _FloatParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if isinstance(df.schema[column].dataType, FloatType):
                return df

            df = df.withColumn(f'{column}_bak', fn.col(column))

            if not isinstance(df.schema[column].dataType, (StringType, NumericType, BooleanType)):
                return df.withColumn(*error_state.add_errors(fn.lit(True), column, details=FLOAT_TYPE))

            return (
                df
                .withColumn(column, fn.col(column).cast('float'))
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column, details=FLOAT_PARSING
                    )
                )
            )

    return _FloatParsingRule()


def str_parsing() -> Rule:
    class _StrParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if isinstance(df.schema[column].dataType, StringType):
                return df

            df = df.withColumn(f'{column}_bak', fn.col(column))

            if not isinstance(df.schema[column].dataType, (NumericType, BooleanType, DateType, TimestampType)):
                return df.withColumn(*error_state.add_errors(fn.lit(True), column, details=STRING_TYPE))

            return df \
                .withColumn(column, fn.col(column).cast('string'))

    return _StrParsingRule()


def bool_parsing() -> Rule:
    class _BoolParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            data_type = df.schema[column].dataType
            if isinstance(data_type, BooleanType):
                return df

            df = df.withColumn(f'{column}_bak', fn.col(column))
            if not isinstance(data_type, (NumericType, StringType)):
                return df.withColumn(*error_state.add_errors(fn.lit(True), column, details=BOOL_TYPE))

            if isinstance(data_type, StringType):
                col = fn.lower(fn.trim(fn.col(column)))
                df = df.withColumn(
                    column, fn.when(col.isin(['on', 'off']), col == 'on').otherwise(fn.col(column).cast('boolean')))
            else:
                df = df.withColumn(column, fn.col(column).cast('boolean'))

            return (
                df
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column, details=BOOL_PARSING
                    )
                )
            )

    return _BoolParsingRule()


def date_parsing() -> Rule:
    class _DateParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if isinstance(df.schema[column].dataType, DateType):
                return df

            df = df.withColumn(f'{column}_bak', fn.col(column))

            if not isinstance(df.schema[column].dataType, (StringType, TimestampType)):
                return df.withColumn(*error_state.add_errors(fn.lit(True), column, details=DATE_TYPE))

            return (
                df
                .withColumn(column, fn.to_date(fn.col(column)))
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column, details=DATE_PARSING
                    )
                )
            )

    return _DateParsingRule()


def datetime_parsing() -> Rule:
    class _DatetimeParsingRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if isinstance(df.schema[column].dataType, TimestampType):
                return df

            df = df.withColumn(f'{column}_bak', fn.col(column))

            if not isinstance(df.schema[column].dataType, (StringType, DateType)):
                return df.withColumn(*error_state.add_errors(fn.lit(True), column, details=DATETIME_TYPE))

            return (
                df
                .withColumn(column, fn.to_timestamp(fn.col(column)))
                .withColumn(
                    *error_state.add_errors(
                        fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column, details=DATETIME_PARSING
                    )
                )
            )

    return _DatetimeParsingRule()


def min_length(value: int) -> Rule:
    """
    Checks if the input has a minimum length.
    """

    if not isinstance(value, int):
        raise ValueError('min_length must be an integer')
    if value < 0:
        raise ValueError('min_length must be a non-negative integer')

    class _MinLengthRule(MaskRule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if not isinstance(df.schema[column].dataType, StringType):
                raise ValueError('min_length rule can only be applied to string columns')

            return super().verify(df, column, error_state)

    return _MinLengthRule(
        lambda col: fn.length(fn.col(col)) < value,
        **STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else '')
    )


def max_length(value: int) -> Rule:
    """
    Checks if the input has a maximum length.
    """

    if not isinstance(value, int):
        raise ValueError('max_length must be an integer')
    if value < 0:
        raise ValueError('max_length must be a non-negative integer')

    class _MaxLengthRule(MaskRule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if not isinstance(df.schema[column].dataType, StringType):
                raise ValueError('max_length rule can only be applied to string columns')

            return super().verify(df, column, error_state)

    return _MaxLengthRule(
        lambda col: fn.length(fn.col(col)) > value,
        **STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else '')
    )


def pattern(regex: str) -> Rule:
    if not isinstance(regex, str):
        raise ValueError('pattern must be a string')
    try:
        re.compile(regex)
    except re.error as e:
        raise ValueError('pattern must be a valid regular expression') from e

    class _PatternRule(MaskRule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if not isinstance(df.schema[column].dataType, StringType):
                raise ValueError('pattern rule can only be applied to string columns')

            return super().verify(df, column, error_state)

    return _PatternRule(
        lambda col: ~fn.col(col).rlike(regex),
        **STRING_PATTERN_MISMATCH.format(pattern=regex)
    )


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    """
    A rule that forbids extra columns in the DataFrame.
    """

    class _ExtraRule(Rule):
        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if column in allowed:
                return df.withColumn(*error_state.add_errors(fn.lit(False), column, details=EXTRA_FORBIDDEN))

            return (
                df
                .withColumnRenamed(column, f'{column}_bak')
                .withColumn(*error_state.add_errors(fn.lit(True), column, details=EXTRA_FORBIDDEN))
            )

    return _ExtraRule()


def isin(values: Iterable[Any]) -> Rule:
    """
    Checks if the input is in a given list of values.
    """

    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return MaskRule(
        lambda col: ~fn.col(col).isin(values),
        **ENUM.format(expected=values)
    )


def notin(values: Iterable[Any]) -> Rule:
    """
    Checks if the input is not in a given list of values.
    """

    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return MaskRule(
        lambda col: fn.col(col).isin(values),
        **NOT_ENUM.format(unexpected=values)
    )
