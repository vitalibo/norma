import inspect
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Union

import pandas as pd

from norma.errors import (
    ENUM,
    EQUAL_TO,
    EXTRA_FORBIDDEN,
    GREATER_THAN,
    GREATER_THAN_EQUAL,
    LESS_THAN,
    LESS_THAN_EQUAL,
    MISSING,
    MULTIPLE_OF,
    NOT_ENUM,
    NOT_EQUAL_TO,
    STRING_PATTERN_MISMATCH,
    STRING_TOO_LONG,
    STRING_TOO_SHORT
)
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    A class that holds the error state of the DataFrame.
    """

    def __init__(self, index: pd.Index) -> None:
        self.masks = defaultdict(lambda: pd.Series(False, index=index))
        self.errors = {}

    def add_errors(self, boolmask: pd.Series, column: str, details: Dict[str, str]) -> None:
        """
        Add errors to the error state.
        """

        for index in boolmask[boolmask].index:
            if index not in self.errors:
                self.errors[index] = {column: {'details': []}}
            elif column not in self.errors[index]:
                self.errors[index][column] = {'details': []}
            self.errors[index][column]['details'].append(details)
        self.masks[column] = self.masks[column] | boolmask.astype(bool)


class MaskRule(Rule):
    """
    The most basic rule type, which takes a boolean mask as input and applies it to the DataFrame to identify errors.
    """

    def __init__(self, boolmask_func: Callable, type: str, msg: str) -> None:  # noqa pylint: disable=redefined-builtin
        self.boolmask_func = boolmask_func
        self.type = type
        self.msg = msg

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        signature = inspect.signature(self.boolmask_func)
        params = {}
        if 'column' in signature.parameters or 'col' in signature.parameters:
            params['col' if 'col' in signature.parameters else 'column'] = column

        boolmask = self.boolmask_func(df, **params)
        error_state.add_errors(boolmask, column, {'type': self.type, 'msg': self.msg})
        return df[column]


class NumberRule(Rule):
    """
    A rule that casts a column to a numeric type.
    In case of casting errors, the rule will add the appropriate error message.
    """

    def __init__(self, dtype: str, type: str, msg: str) -> None:  # noqa pylint: disable=redefined-builtin
        self.dtype = dtype
        self.type = type
        self.msg = msg

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        numeric_series = pd.to_numeric(df[column].convert_dtypes(), errors='coerce').astype(self.dtype)
        boolmask = numeric_series.isna() & df[column].notna()

        error_state.add_errors(boolmask, column, {'type': self.type, 'msg': self.msg})
        return numeric_series


class BooleanRule(Rule):
    """
    A rule that casts a column to a boolean type.
    In case of casting errors, the rule will add the appropriate error message.
    """

    def __init__(self, type: str, msg: str) -> None:  # noqa pylint: disable=redefined-builtin
        self.type = type
        self.msg = msg

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        def replace(regex, value):
            return pd.to_numeric(series.str.replace(regex, value, case=False, regex=True), errors='coerce')

        series = df[column].astype('string')
        true_series = replace(r'^(true|t|yes|y)$', '1')
        false_series = replace(r'^(false|f|no|n)$', '0')
        bool_series = true_series.combine_first(false_series).astype('boolean')
        boolmask = bool_series.isna() & df[column].notna()

        error_state.add_errors(boolmask, column, {'type': self.type, 'msg': self.msg})
        return bool_series


class DatetimeRule(Rule):
    """
    A rule that casts a column to a datetime type.
    """

    # noqa pylint: disable=redefined-builtin
    def __init__(self, dtype: Optional[str], unit: Optional[str], type: str, msg: str) -> None:
        self.dtype = dtype
        self.unit = unit
        self.type = type
        self.msg = msg

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        datetime_series = pd.to_datetime(df[column], unit=self.unit, errors='coerce')
        if self.dtype is not None:
            datetime_series = pd.Series(datetime_series.values.astype(self.dtype), name=column)
        boolmask = datetime_series.isna() & df[column].notna()

        error_state.add_errors(boolmask, column, {'type': self.type, 'msg': self.msg})
        return datetime_series


def required() -> Rule:
    """
    Checks if the input is missing.
    """

    class _RequiredRule(Rule):

        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            if column not in df.columns:
                error_state.add_errors(pd.Series(True, index=df.index), column, MISSING)
                return pd.Series(dtype='string', index=df.index)

            error_state.add_errors(df[column].isna(), column, MISSING)
            return df[column]

    return _RequiredRule()


def equal_to(eq: Any) -> Rule:
    """
    Checks if the input is equal to a given value.
    """

    if eq is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] != eq,
        **EQUAL_TO.format(eq=eq)
    )


def not_equal_to(ne: Any) -> Rule:
    """
    Checks if the input is not equal to a given value.
    """

    if ne is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] == ne,
        **NOT_EQUAL_TO.format(ne=ne)
    )


def greater_than(gt: Any) -> Rule:
    """
    Checks if the input is greater than a given value.
    """

    if gt is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] <= gt,
        **GREATER_THAN.format(gt=gt)
    )


def greater_than_equal(ge: Any) -> Rule:
    """
    Checks if the input is greater than or equal to a given value.
    """

    if ge is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] < ge,
        **GREATER_THAN_EQUAL.format(ge=ge)
    )


def less_than(lt: Any) -> Rule:
    """
    Checks if the input is less than a given value.
    """

    if lt is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] >= lt,
        **LESS_THAN.format(lt=lt)
    )


def less_than_equal(le: Any) -> Rule:
    """
    Checks if the input is less than or equal to a given value.
    """

    if le is None:
        raise ValueError('comparison value must not be None')

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] > le,
        **LESS_THAN_EQUAL.format(le=le)
    )


def multiple_of(multiple: Union[int, float]) -> Rule:
    """
    Checks if the input is a multiple of a given value.
    """
    if multiple is None:
        raise ValueError('multiple_of must not be None')
    if not isinstance(multiple, (int, float)):
        raise ValueError('multiple_of must be an integer or a float')

    class _MultipleOfRule(MaskRule):
        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError('multiple_of rule can only be applied to numeric columns')

            return super().verify(df, column, error_state)

    return _MultipleOfRule(
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        lambda df, col: df[col][df[col].notna()] % multiple != 0,
        **MULTIPLE_OF.format(multiple_of=multiple))


def int_parsing() -> Rule:
    """
    Checks if the input can be parsed as an integer.
    The rule modifies the original column to cast it to an integer type.
    """

    return NumberRule(
        dtype='Int64',
        type='int_parsing',
        msg='Input should be a valid integer, unable to parse value as an integer'
    )


def float_parsing() -> Rule:
    """
    Checks if the input can be parsed as a float.
    The rule modifies the original column to cast it to a float type.
    """

    return NumberRule(
        dtype='Float64',
        type='float_parsing',
        msg='Input should be a valid float, unable to parse value as a float'
    )


def str_parsing() -> Rule:
    """
    The rule modifies the original column to cast it to a string type.
    """

    class _StringRule(Rule):
        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            return df[column].astype('string')

    return _StringRule()


def bool_parsing() -> Rule:
    """
    Checks if the input can be parsed as a boolean.
    The rule modifies the original column to cast it to a boolean type.
    """

    return BooleanRule(
        type='boolean_parsing',
        msg='Input should be a valid boolean, unable to parse value as a boolean'
    )


def datetime_parsing() -> Rule:
    """
    Checks if the input can be parsed as a datetime.
    """

    return DatetimeRule(
        dtype=None, unit=None,
        type='datetime_parsing',
        msg='Input should be a valid datetime, unable to parse value as a datetime'
    )


def timestamp_parsing(unit: str = 's') -> Rule:
    """
    Checks if the input can be parsed as a datetime.
    """

    if unit not in ['s', 'ms', 'us', 'ns']:
        raise ValueError('unit should be one of "s", "ms", "us", "ns"')

    return DatetimeRule(
        dtype=f'datetime64[{unit}]', unit=unit,
        type='timestamp_parsing',
        msg='Input should be a valid epoch timestamp, unable to parse value as a epoch timestamp'
    )


def date_parsing() -> Rule:
    """
    Checks if the input can be parsed as a date.
    """

    return DatetimeRule(
        dtype='datetime64[D]', unit=None,
        type='date_parsing',
        msg='Input should be a valid date, unable to parse value as a date'
    )


def time_parsing() -> Rule:
    """
    Checks if the input can be parsed as a time.
    """

    class _TimeRule(Rule):
        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            time_s, time_ms, time_s_tz, time_ms_tz = (
                pd.to_datetime(df[column], errors='coerce', format=fstr)
                for fstr in ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M:%S%z', '%H:%M:%S.%f%z']
            )

            time_tz = time_s_tz.combine_first(time_ms_tz)
            has_series_tz = time_tz.notna().any()
            strformat = '%H:%M:%S.%f+0000' if has_series_tz else '%H:%M:%S.%f'
            time_series = time_s.combine_first(time_ms).dt.strftime(strformat).astype('string')

            if has_series_tz:
                time_tz = time_tz.map(lambda x: x.strftime('%H:%M:%S.%f%z') if pd.notnull(x) else None)
                time_series = time_series.combine_first(time_tz.astype('string'))

            boolmask = time_series.isna() & df[column].notna()
            error_state.add_errors(
                boolmask, column, {
                    'type': 'time_parsing',
                    'msg': 'Input should be a valid time, unable to parse value as a time'
                }
            )

            return time_series

    return _TimeRule()


def min_length(value: int) -> Rule:
    """
    Checks if the input has a minimum length.
    """

    if not isinstance(value, int):
        raise ValueError('min_length must be an integer')
    if value < 0:
        raise ValueError('min_length must be a non-negative integer')

    class _MinLengthRule(MaskRule):
        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            if not pd.api.types.is_string_dtype(df[column]):
                raise ValueError('min_length rule can only be applied to string columns')

            return super().verify(df, column, error_state)

    return _MinLengthRule(
        lambda df, col: df[col][df[col].notna()].str.len() < value,
        **STRING_TOO_SHORT.format(min_length=value, _expected_plural_='s' if value > 1 else '')
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
        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            if not pd.api.types.is_string_dtype(df[column]):
                raise ValueError('max_length rule can only be applied to string columns')

            return super().verify(df, column, error_state)

    return _MaxLengthRule(
        lambda df, col: df[col][df[col].notna()].str.len() > value,
        **STRING_TOO_LONG.format(max_length=value, _expected_plural_='s' if value > 1 else '')
    )


def pattern(regex: str) -> Rule:
    """
    Checks if the input matches a given regex pattern.
    """

    if not isinstance(regex, str):
        raise ValueError('pattern must be a string')
    try:
        re.compile(regex)
    except re.error as e:
        raise ValueError('pattern must be a valid regular expression') from e

    class _PatternRule(MaskRule):
        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            if not pd.api.types.is_string_dtype(df[column]):
                raise ValueError('pattern rule can only be applied to string columns')

            return super().verify(df, column, error_state)

    return _PatternRule(
        lambda df, col: ~df[col][df[col].notna()].str.match(regex, na=False),
        **STRING_PATTERN_MISMATCH.format(pattern=regex)
    )


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    """
    A rule that forbids extra columns in the DataFrame.
    """

    class _ExtraRule(Rule):

        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> Optional[pd.Series]:
            if column in allowed:
                return df[column]

            error_state.add_errors(
                pd.Series(True, index=df.index), column,
                EXTRA_FORBIDDEN
            )

            del error_state.masks[column]
            df.drop(column, axis=1, inplace=True)
            return None

    return _ExtraRule()


def isin(values: Iterable[Any]) -> Rule:
    """
    Checks if the input is in a given list of values.
    """

    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return MaskRule(
        lambda df, col: ~df[col][df[col].notna()].isin(values),
        **ENUM.format(expected=values)
    )


def notin(values: Iterable[Any]) -> Rule:
    """
    Checks if the input is not in a given list of values.
    """

    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return MaskRule(
        lambda df, col: df[col][df[col].notna()].isin(values),
        **NOT_ENUM.format(unexpected=values)
    )
