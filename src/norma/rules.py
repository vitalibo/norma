import inspect
from collections import defaultdict
from typing import Any, Callable, Dict, Union

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

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
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


def required() -> Rule:
    """
    Checks if the input is missing.
    """

    return MaskRule(
        lambda df, col: df[col].isna(),
        error_type='missing',
        error_msg='Field required')


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


def min_length(value: int) -> Rule:
    """
    Checks if the input has a minimum length.
    """

    return MaskRule(
        lambda df, col: df[col].str.len() < value,
        error_type='min_length',
        error_msg=f'Input should have a minimum length of {value}'
    )


def max_length(value: int) -> Rule:
    """
    Checks if the input has a maximum length.
    """

    return MaskRule(
        lambda df, col: df[col].str.len() > value,
        error_type='max_length',
        error_msg=f'Input should have a maximum length of {value}'
    )


def pattern(regex: str) -> Rule:
    """
    Checks if the input matches a given regex pattern.
    """

    return MaskRule(
        lambda df, col: ~df[col].str.match(regex, na=False),
        error_type='pattern',
        error_msg=f'Input should match the pattern {regex}'
    )
