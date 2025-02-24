import abc
import inspect
import re
from typing import Any, Iterable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import BooleanType, DateType, FloatType, IntegerType, NumericType, StringType, TimestampType

from norma import errors
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    A class to represent the state of errors in a DataFrame.
    """

    def __init__(self, error_column: str):
        self.error_column = error_column

    def add_errors(self, boolmask: Column, column: str, **kwargs):
        details = kwargs.get('details') or kwargs
        details_col = fn.when(boolmask, fn.struct(*[fn.lit(v).alias(k) for k, v in details.items()]))

        name = f'{self.error_column}_{column}'
        return lambda df: df.withColumn(name, fn.array_append(fn.col(name), details_col))


class MaskRule(Rule):
    """
    A rule to validate a column in a DataFrame.
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        signature = inspect.signature(self.func)
        if 'df' not in signature.parameters:
            return df.transform(error_state.add_errors(self.func(column), column, **self.kwargs))

        params = {}
        if set(signature.parameters) & {'col', 'column'}:
            params['col' if 'col' in signature.parameters else 'column'] = column
        if 'error_state' in signature.parameters:
            params['error_state'] = error_state
        return self.func(df, **params)

    def compose(self, before) -> Rule:
        return MaskRule(
            lambda df, col, error_state: self.verify(before(df, col, error_state), col, error_state), **self.kwargs
        )


def rule(func, **kwargs) -> MaskRule:
    return MaskRule(func, **kwargs)


def required() -> Rule:
    @Rule.new
    def verify(df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        mask = fn.col(column).isNull()
        if column not in df.columns:
            df = df.withColumn(column, fn.lit(None))
            mask = fn.lit(True)

        return df.transform(error_state.add_errors(mask, column, details=errors.MISSING))

    return verify


def equal_to(eq: Any) -> Rule:
    if eq is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda col: fn.col(col) != fn.lit(eq),
        details=errors.EQUAL_TO.format(eq=eq)
    )


def not_equal_to(ne: Any) -> Rule:
    if ne is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda col: fn.col(col) == fn.lit(ne),
        details=errors.NOT_EQUAL_TO.format(ne=ne)
    )


def greater_than(gt: Any) -> Rule:
    if gt is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda col: fn.col(col) <= fn.lit(gt),
        details=errors.GREATER_THAN.format(gt=gt)
    )


def greater_than_equal(ge: Any) -> Rule:
    if ge is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda col: fn.col(col) < fn.lit(ge),
        details=errors.GREATER_THAN_EQUAL.format(ge=ge)
    )


def less_than(lt: Any) -> Rule:
    if lt is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda col: fn.col(col) >= fn.lit(lt),
        details=errors.LESS_THAN.format(lt=lt)
    )


def less_than_equal(le: Any) -> Rule:
    if le is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda col: fn.col(col) > fn.lit(le),
        details=errors.LESS_THAN_EQUAL.format(le=le)
    )


def multiple_of(multiple: Any) -> Rule:
    if multiple is None:
        raise ValueError('multiple_of must not be None')
    if not isinstance(multiple, (int, float)):
        raise ValueError('multiple_of must be an integer or a float')

    def before(df, col, _):
        if not isinstance(df.schema[col].dataType, NumericType):
            raise ValueError('multiple_of rule can only be applied to numeric columns')
        return df

    return rule(
        lambda col: (fn.col(col) % fn.lit(multiple)) != fn.lit(0),
        details=errors.MULTIPLE_OF.format(multiple_of=multiple)
    ).compose(before)


def min_length(value: int) -> Rule:
    if not isinstance(value, int):
        raise ValueError('min_length must be an integer')
    if value < 0:
        raise ValueError('min_length must be a non-negative integer')

    def before(df, column, _):
        if not isinstance(df.schema[column].dataType, StringType):
            raise ValueError('min_length rule can only be applied to string columns')
        return df

    return rule(
        lambda col: fn.length(fn.col(col)) < value,
        details=errors.STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else '')
    ).compose(before)


def max_length(value: int) -> Rule:
    if not isinstance(value, int):
        raise ValueError('max_length must be an integer')
    if value < 0:
        raise ValueError('max_length must be a non-negative integer')

    def before(df, column, _):
        if not isinstance(df.schema[column].dataType, StringType):
            raise ValueError('max_length rule can only be applied to string columns')
        return df

    return rule(
        lambda col: fn.length(fn.col(col)) > value,
        details=errors.STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else '')
    ).compose(before)


def pattern(regex: str) -> Rule:
    if not isinstance(regex, str):
        raise ValueError('pattern must be a string')
    try:
        re.compile(regex)
    except re.error as e:
        raise ValueError('pattern must be a valid regular expression') from e

    def before(df, column, _):
        if not isinstance(df.schema[column].dataType, StringType):
            raise ValueError('pattern rule can only be applied to string columns')
        return df

    return rule(
        lambda col: ~fn.col(col).rlike(regex),
        details=errors.STRING_PATTERN_MISMATCH.format(pattern=regex)
    ).compose(before)


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    @Rule.new
    def verify(df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        if column in allowed:
            return df

        return df \
            .withColumnRenamed(column, f'{column}_bak') \
            .transform(error_state.add_errors(fn.lit(True), column, details=errors.EXTRA_FORBIDDEN))

    return verify


def isin(values: Iterable[Any]) -> Rule:
    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return rule(
        lambda col: ~fn.col(col).isin(values),
        details=errors.ENUM.format(expected=values)
    )


def notin(values: Iterable[Any]) -> Rule:
    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return rule(
        lambda col: fn.col(col).isin(values),
        details=errors.NOT_ENUM.format(unexpected=values)
    )


def int_parsing() -> Rule:
    return NumericTypeRule(IntegerType, errors.INT_TYPE, errors.INT_PARSING)


def float_parsing():
    return NumericTypeRule(FloatType, errors.FLOAT_TYPE, errors.FLOAT_PARSING)


def str_parsing() -> Rule:
    return StringTypeRule()


def bool_parsing() -> Rule:
    return BooleanTypeRule()


def date_parsing() -> Rule:
    return DateTypeRule()


def datetime_parsing() -> Rule:
    return DatetimeTypeRule()


class DataTypeRule(Rule):
    """
    A rule to validate the data type of column in a DataFrame.
    """

    def __init__(self, dtype, unsupported, details):
        self.dtype = dtype
        self.unsupported = unsupported
        self.details = details

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        if isinstance(df.schema[column].dataType, self.dtype):
            return df

        if df.schema[column].dataType.typeName() == 'void':
            return df.withColumn(column, fn.col(column).cast(self.dtype.typeName()))

        df = df.withColumn(f'{column}_bak', fn.col(column))
        if not isinstance(df.schema[column].dataType, self.unsupported):
            return df.transform(error_state.add_errors(fn.lit(True), column, details=self.details))

        return self.cast(df, column, error_state)

    @abc.abstractmethod
    def cast(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        """
        Cast the column to the appropriate data type.
        """


class NumericTypeRule(DataTypeRule):
    """
    A rule to validate the numeric data type of column in a DataFrame.
    """

    def __init__(self, dtype, numeric_type, numeric_parsing):
        super().__init__(dtype, (StringType, NumericType, BooleanType), numeric_type)
        self.numeric_parsing = numeric_parsing

    def cast(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        return df \
            .withColumn(column, fn.col(column).cast(self.dtype.typeName())) \
            .transform(error_state.add_errors(fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column,
                                              details=self.numeric_parsing))


class StringTypeRule(DataTypeRule):
    """
    A rule to validate the string data type of column in a DataFrame.
    """

    def __init__(self):
        super().__init__(StringType, (NumericType, BooleanType, DateType, TimestampType), errors.STRING_TYPE)

    def cast(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        return df.withColumn(column, fn.col(column).cast('string'))


class BooleanTypeRule(DataTypeRule):
    """
    A rule to validate the boolean data type of column in a DataFrame.
    """

    def __init__(self):
        super().__init__(BooleanType, (NumericType, StringType), errors.BOOL_TYPE)

    def cast(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        expr = fn.col(column)
        if isinstance(df.schema[column].dataType, StringType):
            expr = fn.lower(fn.trim(fn.col(column)))
            expr = fn.when(expr.isin(['on', 'off']), expr == 'on').otherwise(fn.col(column).cast('boolean'))

        return df \
            .withColumn(column, expr.cast('boolean')) \
            .transform(error_state.add_errors(fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column,
                                              details=errors.BOOL_PARSING))


class DatetimeTypeRule(DataTypeRule):
    """
    A rule to validate the datetime data type of column in a DataFrame.
    """

    def __init__(self):
        super().__init__(TimestampType, (StringType, DateType), errors.DATETIME_TYPE)

    def cast(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        return df \
            .withColumn(column, fn.to_timestamp(fn.col(column))) \
            .transform(error_state.add_errors(fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column,
                                              details=errors.DATETIME_PARSING))


class DateTypeRule(DataTypeRule):
    """
    A rule to validate the date data type of column in a DataFrame.
    """

    def __init__(self):
        super().__init__(DateType, (StringType, TimestampType), errors.DATE_TYPE)

    def cast(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        return df \
            .withColumn(column, fn.to_date(fn.col(column))) \
            .transform(error_state.add_errors(fn.col(column).isNull() & fn.col(f'{column}_bak').isNotNull(), column,
                                              details=errors.DATE_PARSING))
