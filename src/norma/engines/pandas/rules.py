import inspect
import re
from collections import defaultdict
from typing import Any, Iterable, Optional, Union

import pandas as pd

from norma import errors
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    A class that holds the error state of the DataFrame.
    """

    def __init__(self, index: pd.Index) -> None:
        self.masks = defaultdict(lambda: pd.Series(False, index=index))
        self.errors = {}

    def add_errors(self, boolmask, column, **kwargs):
        """
        Add errors to the error state.
        """

        for index in boolmask[boolmask].index:
            if index not in self.errors:
                self.errors[index] = {column: {'details': []}}
            elif column not in self.errors[index]:
                self.errors[index][column] = {'details': []}
            self.errors[index][column]['details'].append(dict(kwargs.get('details', {}) or kwargs))
        self.masks[column] = self.masks[column] | boolmask.astype(bool)


class MaskRule(Rule):
    """
    The most basic rule type, which takes a boolean mask as input and applies it to the DataFrame to identify errors.
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        def inspect_params(f):
            signature = inspect.signature(f)
            params = {}
            if set(signature.parameters) & {'col', 'column'}:
                params['col' if 'col' in signature.parameters else 'column'] = column
            if 'error_state' in signature.parameters:
                params['error_state'] = error_state
            return params

        if 'pre_func' in self.kwargs:
            pre_func = self.kwargs['pre_func']
            df = pre_func(df, **inspect_params(pre_func))

        boolmask = self.func(df, **inspect_params(self.func))
        error_state.add_errors(boolmask, column, details=self.kwargs.get('details', {}) or self.kwargs)
        return df[column]


def rule(func, **kwargs) -> MaskRule:
    return MaskRule(func, **kwargs)


def required() -> Rule:
    @Rule.new
    def verify(df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        if column not in df.columns:
            error_state.add_errors(pd.Series(True, index=df.index), column, details=errors.MISSING)
            return pd.Series(dtype='string', index=df.index)

        error_state.add_errors(df[column].isna(), column, details=errors.MISSING)
        return df[column]

    return verify


def equal_to(eq: Any) -> Rule:
    if eq is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda df, col: df[col][df[col].notna()] != eq,
        details=errors.EQUAL_TO.format(eq=eq)
    )


def not_equal_to(ne: Any) -> Rule:
    if ne is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda df, col: df[col][df[col].notna()] == ne,
        details=errors.NOT_EQUAL_TO.format(ne=ne)
    )


def greater_than(gt: Any) -> Rule:
    if gt is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda df, col: df[col][df[col].notna()] <= gt,
        details=errors.GREATER_THAN.format(gt=gt)
    )


def greater_than_equal(ge: Any) -> Rule:
    if ge is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda df, col: df[col][df[col].notna()] < ge,
        details=errors.GREATER_THAN_EQUAL.format(ge=ge)
    )


def less_than(lt: Any) -> Rule:
    if lt is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda df, col: df[col][df[col].notna()] >= lt,
        details=errors.LESS_THAN.format(lt=lt)
    )


def less_than_equal(le: Any) -> Rule:
    if le is None:
        raise ValueError('comparison value must not be None')

    return rule(
        lambda df, col: df[col][df[col].notna()] > le,
        details=errors.LESS_THAN_EQUAL.format(le=le)
    )


def multiple_of(multiple: Union[int, float]) -> Rule:
    if multiple is None:
        raise ValueError('multiple_of must not be None')
    if not isinstance(multiple, (int, float)):
        raise ValueError('multiple_of must be an integer or a float')

    def before(df, column):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError('multiple_of rule can only be applied to numeric columns')
        return df

    return rule(
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        lambda df, col: df[col][df[col].notna()] % multiple != 0.0,
        details=errors.MULTIPLE_OF.format(multiple_of=multiple),
        pre_func=before
    )


def int_parsing() -> Rule:
    return NumberTypeRule('Int64', errors.INT_TYPE, errors.INT_PARSING)


def float_parsing() -> Rule:
    return NumberTypeRule('Float64', errors.FLOAT_TYPE, errors.FLOAT_PARSING)


def str_parsing() -> Rule:
    return StringTypeRule()


def bool_parsing() -> Rule:
    return BooleanTypeRule()


def datetime_parsing() -> Rule:
    return DatetimeTypeRule(None, errors.DATETIME_TYPE, errors.DATETIME_PARSING)


def date_parsing() -> Rule:
    return DatetimeTypeRule('datetime64[D]', errors.DATE_TYPE, errors.DATE_PARSING)


def min_length(value: int) -> Rule:
    if not isinstance(value, int):
        raise ValueError('min_length must be an integer')
    if value < 0:
        raise ValueError('min_length must be a non-negative integer')

    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('min_length rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: df[col][df[col].notna()].str.len() < value,
        details=errors.STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else ''),
        pre_func=before
    )


def max_length(value: int) -> Rule:
    if not isinstance(value, int):
        raise ValueError('max_length must be an integer')
    if value < 0:
        raise ValueError('max_length must be a non-negative integer')

    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('max_length rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: df[col][df[col].notna()].str.len() > value,
        details=errors.STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else ''),
        pre_func=before
    )


def pattern(regex: str) -> Rule:
    if not isinstance(regex, str):
        raise ValueError('pattern must be a string')
    try:
        re.compile(regex)
    except re.error as e:
        raise ValueError('pattern must be a valid regular expression') from e

    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('pattern rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: ~df[col][df[col].notna()].str.match(regex, na=False),
        details=errors.STRING_PATTERN_MISMATCH.format(pattern=regex),
        pre_func=before
    )


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    @Rule.new
    def verify(df: pd.DataFrame, column: str, error_state: ErrorState) -> Optional[pd.Series]:
        if column in allowed:
            return df[column]

        error_state.add_errors(pd.Series(True, index=df.index), column, details=errors.EXTRA_FORBIDDEN)

        del error_state.masks[column]
        df.drop(column, axis=1, inplace=True)
        return None

    return verify


def isin(values: Iterable[Any]) -> Rule:
    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return rule(
        lambda df, col: ~df[col][df[col].notna()].isin(values),
        details=errors.ENUM.format(expected=values)
    )


def notin(values: Iterable[Any]) -> Rule:
    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return rule(
        lambda df, col: df[col][df[col].notna()].isin(values),
        details=errors.NOT_ENUM.format(unexpected=values)
    )


class NumberTypeRule(Rule):
    """
    A rule that verifies the numeric type of the column.
    """

    def __init__(self, dtype, numeric_type, numeric_parsing):
        self.dtype = dtype
        self.numeric_type = numeric_type
        self.numeric_parsing = numeric_parsing

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        if df[column].dtype == self.dtype:
            return df[column]

        non_parsing_type_series = pd.Series(False, index=df.index)
        if not (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_numeric_dtype(df[column])):
            if not pd.api.types.is_object_dtype(df[column]):
                error_state.add_errors(pd.Series(True, index=df.index), column, details=self.numeric_type)
                return pd.Series(dtype=self.dtype, name=column, index=df.index)

            non_parsing_type_series = df[column].apply(lambda x: not isinstance(x, (str, bool, int, float)))
            error_state.add_errors(non_parsing_type_series, column, details=self.numeric_type)

        numeric_series = pd.to_numeric(df[column].convert_dtypes(), errors='coerce').astype(self.dtype)

        boolmask = numeric_series.isna() & df[column].notna() & ~non_parsing_type_series
        error_state.add_errors(boolmask, column, details=self.numeric_parsing)
        return numeric_series


class BooleanTypeRule(Rule):
    """
    A rule that verifies the boolean type of the column.
    """

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        if pd.api.types.is_bool_dtype(df[column]):
            return df[column].astype('boolean')

        non_parsing_type_series = pd.Series(False, index=df.index)
        if not (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_numeric_dtype(df[column])):
            if not pd.api.types.is_object_dtype(df[column]):
                error_state.add_errors(pd.Series(True, index=df.index), column, details=errors.BOOL_TYPE)
                return pd.Series(dtype='boolean', name=column, index=df.index)

            non_parsing_type_series = df[column].apply(lambda x: not isinstance(x, (str, bool, int, float)))
            error_state.add_errors(non_parsing_type_series, column, details=errors.BOOL_TYPE)

        def replace_str(regex, value):
            return pd.to_numeric(series.str.replace(regex, value, case=False, regex=True), errors='coerce')

        series = df[column].astype('string')
        true_series = replace_str(r'^\s*(true|t|yes|y|on)\s*$', '1')
        false_series = replace_str(r'^\s*(false|f|no|n|off)\s*$', '0')
        bool_series = true_series.combine_first(false_series).astype('boolean')

        boolmask = bool_series.isna() & df[column].notna() & ~non_parsing_type_series
        error_state.add_errors(boolmask, column, details=errors.BOOL_PARSING)
        return bool_series


class StringTypeRule(Rule):
    """
    A rule that verifies the string type of the column.
    """

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        if df[column].dtype == 'string[python]':
            return df[column]

        non_parsing_type_series = pd.Series(False, index=df.index)
        bool_series = pd.Series(False, index=df.index)
        if pd.api.types.is_object_dtype(df[column]):
            non_parsing_type_series = df[column].apply(lambda x: not isinstance(x, (str, bool, int, float)))
            error_state.add_errors(non_parsing_type_series, column, details=errors.STRING_TYPE)
            bool_series = df[column].apply(lambda x: isinstance(x, bool))

        if pd.api.types.is_bool_dtype(df[column]):
            str_series = df[column].astype('string').str.lower()
        else:
            str_series = df[column].astype('string')
            str_series[bool_series] = str_series[bool_series].str.lower()

        str_series[non_parsing_type_series] = None
        return str_series


class DatetimeTypeRule(Rule):
    """
    A rule that verifies the datetime type of the column.
    """

    def __init__(self, dtype, dt_type, dt_parsing):
        self.dtype = dtype
        self.dt_type = dt_type
        self.dt_parsing = dt_parsing

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        if df[column].dtype == self.dtype:
            return df[column]

        non_parsing_type_series = pd.Series(False, index=df.index)
        if not (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_datetime64_any_dtype(df[column])):
            if not pd.api.types.is_object_dtype(df[column]):
                error_state.add_errors(pd.Series(True, index=df.index), column, details=self.dt_type)
                return pd.Series(dtype=self.dtype or 'datetime64[ns]', name=column, index=df.index)

            non_parsing_type_series = df[column].apply(lambda x: not isinstance(x, str))
            error_state.add_errors(non_parsing_type_series & df[column].notna(), column, details=self.dt_type)

        datetime_series = pd.to_datetime(df[column], errors='coerce', utc=True)
        if self.dtype is not None:
            datetime_series = pd.Series(datetime_series.values.astype(self.dtype), name=column)

        boolmask = datetime_series.isna() & df[column].notna() & ~non_parsing_type_series
        error_state.add_errors(boolmask, column, details=self.dt_parsing)
        return datetime_series
