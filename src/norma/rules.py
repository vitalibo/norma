import inspect
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Union

import pandas as pd

__all__ = [
    'MaskRule',
    'required',
    'equal_to',
    'eq',
    'not_equal_to',
    'ne',
    'greater_than',
    'gt',
    'greater_than_equal',
    'ge',
    'less_than',
    'lt',
    'less_than_equal',
    'le',
    'multiple_of',
    'int_parsing',
    'float_parsing',
    'string_parsing',
    'boolean_parsing',
    'min_length',
    'max_length',
    'pattern',
]


class ErrorState:
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


class Rule:
    """
    Defines the interface for a rule that can be applied to a DataFrame.
    """

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> Optional[pd.Series]:
        """
        Verify the DataFrame and return a validated Series.
        """


class MaskRule(Rule):
    """
    The most basic rule type, which takes a boolean mask as input and applies it to the DataFrame to identify errors.
    """

    def __init__(self, boolmask_func: Callable, error_type: str, error_msg: str) -> None:
        self.boolmask_func = boolmask_func
        self.error_type = error_type
        self.error_msg = error_msg

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        signature = inspect.signature(self.boolmask_func)
        params = {}
        if 'column' in signature.parameters or 'col' in signature.parameters:
            params['col' if 'col' in signature.parameters else 'column'] = column

        boolmask = self.boolmask_func(df, **params)
        error_state.add_errors(boolmask, column, {'type': self.error_type, 'msg': self.error_msg})
        return df[column]


class NumberRule(Rule):
    """
    A rule that casts a column to a numeric type.
    In case of casting errors, the rule will add the appropriate error message.
    """

    def __init__(self, dtype: str, error_type: str, error_msg: str) -> None:
        self.dtype = dtype
        self.error_type = error_type
        self.error_msg = error_msg

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        numeric_series = pd.to_numeric(df[column].convert_dtypes(), errors='coerce').astype(self.dtype)
        boolmask = numeric_series.isna() & df[column].notna()

        error_state.add_errors(boolmask, column, {'type': self.error_type, 'msg': self.error_msg})
        return numeric_series


class BooleanRule(Rule):
    """
    A rule that casts a column to a boolean type.
    In case of casting errors, the rule will add the appropriate error message.
    """

    def __init__(self, error_type: str, error_msg: str) -> None:
        self.error_type = error_type
        self.error_msg = error_msg

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        def replace(regex, value):
            return pd.to_numeric(series.str.replace(regex, value, case=False, regex=True), errors='coerce')

        series = df[column].astype('string')
        true_series = replace(r'^(true|t|yes|y)$', '1')
        false_series = replace(r'^(false|f|no|n)$', '0')
        bool_series = true_series.combine_first(false_series).astype('boolean')
        boolmask = bool_series.isna() & df[column].notna()

        error_state.add_errors(boolmask, column, {'type': self.error_type, 'msg': self.error_msg})
        return bool_series


class DatetimeRule(Rule):
    """
    A rule that casts a column to a datetime type.
    """

    def __init__(self, dtype: Optional[str], unit: Optional[str], error_type: str, error_msg: str) -> None:
        self.dtype = dtype
        self.unit = unit
        self.error_type = error_type
        self.error_msg = error_msg

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        datetime_series = pd.to_datetime(df[column], unit=self.unit, errors='coerce')
        if self.dtype is not None:
            datetime_series = pd.Series(datetime_series.values.astype(self.dtype), name=column)
        boolmask = datetime_series.isna() & df[column].notna()

        error_state.add_errors(boolmask, column, {'type': self.error_type, 'msg': self.error_msg})
        return datetime_series


def required() -> Rule:
    """
    Checks if the input is missing.
    """

    class _RequiredRule(Rule):

        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            details = {'type': 'missing', 'msg': 'Field required'}

            if column not in df.columns:
                error_state.add_errors(pd.Series(True, index=df.index), column, details)
                return pd.Series(dtype='string', index=df.index)

            error_state.add_errors(df[column].isna(), column, details)
            return df[column]

    return _RequiredRule()


def equal_to(value: Any) -> Rule:
    """
    Checks if the input is equal to a given value.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] != value,
        error_type='equal_to',
        error_msg=f'Input should be equal to {value}')


def eq(value: Any) -> Rule:
    """
    Alias for equal_to.
    """

    return equal_to(value)


def not_equal_to(value: Any) -> Rule:
    """
    Checks if the input is not equal to a given value.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] == value,
        error_type='not_equal_to',
        error_msg=f'Input should not be equal to {value}')


def ne(value: Any) -> Rule:
    """
    Alias for not_equal_to.
    """

    return not_equal_to(value)


def greater_than(value: Any) -> Rule:
    """
    Checks if the input is greater than a given value.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] <= value,
        error_type='greater_than',
        error_msg=f'Input should be greater than {value}')


def gt(value: Any) -> Rule:
    """
    Alias for greater_than.
    """

    return greater_than(value)


def greater_than_equal(value: Any) -> Rule:
    """
    Checks if the input is greater than or equal to a given value.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] < value,
        error_type='greater_than_equal',
        error_msg=f'Input should be greater than or equal to {value}')


def ge(value: Any) -> Rule:
    """
    Alias for greater_than_equal.
    """

    return greater_than_equal(value)


def less_than(value: Any) -> Rule:
    """
    Checks if the input is less than a given value.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] >= value,
        error_type='less_than',
        error_msg=f'Input should be less than {value}')


def lt(value: Any) -> Rule:
    """
    Alias for less_than.
    """

    return less_than(value)


def less_than_equal(value: Any) -> Rule:
    """
    Checks if the input is less than or equal to a given value.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()] > value,
        error_type='less_than_equal',
        error_msg=f'Input should be less than or equal to {value}')


def le(value: Any) -> Rule:
    """
    Alias for less_than_equal.
    """

    return less_than_equal(value)


def multiple_of(value: Union[int, float]) -> Rule:
    """
    Checks if the input is a multiple of a given value.
    """

    return MaskRule(
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        lambda df, col: df[col][df[col].notna()] % value != 0,
        error_type='multiple_of',
        error_msg=f'Input should be a multiple of {value}')


def int_parsing():
    """
    Checks if the input can be parsed as an integer.
    The rule modifies the original column to cast it to an integer type.
    """

    return NumberRule(
        dtype='Int64',
        error_type='int_parsing',
        error_msg='Input should be a valid integer, unable to parse value as an integer'
    )


def float_parsing():
    """
    Checks if the input can be parsed as a float.
    The rule modifies the original column to cast it to a float type.
    """

    return NumberRule(
        dtype='Float64',
        error_type='float_parsing',
        error_msg='Input should be a valid float, unable to parse value as a float'
    )


def string_parsing():
    """
    The rule modifies the original column to cast it to a string type.
    """

    class _StringRule(Rule):
        def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
            return df[column].astype('string')

    return _StringRule()


def boolean_parsing():
    """
    Checks if the input can be parsed as a boolean.
    The rule modifies the original column to cast it to a boolean type.
    """

    return BooleanRule(
        error_type='boolean_parsing',
        error_msg='Input should be a valid boolean, unable to parse value as a boolean'
    )


def datetime_parsing() -> Rule:
    """
    Checks if the input can be parsed as a datetime.
    """

    return DatetimeRule(
        dtype=None, unit=None,
        error_type='datetime_parsing',
        error_msg='Input should be a valid datetime, unable to parse value as a datetime'
    )


def timestamp_parsing(unit: str = 's') -> Rule:
    """
    Checks if the input can be parsed as a datetime.
    """

    if unit not in ['s', 'ms', 'us', 'ns']:
        raise ValueError('unit should be one of "s", "ms", "us", "ns"')

    return DatetimeRule(
        dtype=f'datetime64[{unit}]', unit=unit,
        error_type='timestamp_parsing',
        error_msg='Input should be a valid epoch timestamp, unable to parse value as a epoch timestamp'
    )


def date_parsing() -> Rule:
    """
    Checks if the input can be parsed as a date.
    """

    return DatetimeRule(
        dtype='datetime64[D]', unit=None,
        error_type='date_parsing',
        error_msg='Input should be a valid date, unable to parse value as a date'
    )


def min_length(value: int) -> Rule:
    """
    Checks if the input has a minimum length.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()].str.len() < value,
        error_type='min_length',
        error_msg=f'Input should have a minimum length of {value}'
    )


def max_length(value: int) -> Rule:
    """
    Checks if the input has a maximum length.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()].str.len() > value,
        error_type='max_length',
        error_msg=f'Input should have a maximum length of {value}'
    )


def pattern(regex: str) -> Rule:
    """
    Checks if the input matches a given regex pattern.
    """

    return MaskRule(
        lambda df, col: ~df[col][df[col].notna()].str.match(regex, na=False),
        error_type='pattern',
        error_msg=f'Input should match the pattern {regex}'
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
                {'type': 'extra_forbidden', 'msg': 'Extra inputs are not permitted'}
            )

            del error_state.masks[column]
            df.drop(column, axis=1, inplace=True)
            return None

    return _ExtraRule()


def isin(values: Iterable[Any]) -> Rule:
    """
    Checks if the input is in a given list of values.
    """

    return MaskRule(
        lambda df, col: ~df[col][df[col].notna()].isin(values),
        error_type='isin',
        error_msg=f'Input should be in {values}'
    )


def notin(values: Iterable[Any]) -> Rule:
    """
    Checks if the input is not in a given list of values.
    """

    return MaskRule(
        lambda df, col: df[col][df[col].notna()].isin(values),
        error_type='notin',
        error_msg=f'Input should not be in {values}'
    )
