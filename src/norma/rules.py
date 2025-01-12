import inspect
from collections import defaultdict
from typing import Callable, Dict, Any

import pandas as pd


class ErrorState:
    """
    A class that holds the error state of the DataFrame.
    """

    def __init__(self, index: pd.Index) -> None:
        self.errors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.masks = defaultdict(lambda: pd.Series(False, index=index))

    def add_errors(self, boolmask: pd.Series, column: str, details: Dict[str, str]) -> None:
        """
        Add errors to the error state.
        """

        for index in boolmask[boolmask].index:
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


def multiple_of(value: int | float) -> Rule:
    """
    Checks if the input is a multiple of a given value.
    """

    return MaskRule(
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
