import abc
import inspect
from collections import defaultdict
from typing import Any, Iterable, Optional, Union
from uuid import UUID

import pandas as pd

from norma import errors
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    Error state for Pandas DataFrame validation

    :param index: The index of the original DataFrame
    """

    def __init__(self, index: pd.Index) -> None:
        self.masks = defaultdict(lambda: pd.Series(False, index=index))
        self.errors = {}

    def add_errors(self, boolmask, column, **kwargs):
        """
        Add errors to the error state for a given column
        """

        for index in boolmask[boolmask].index:
            if index not in self.errors:
                self.errors[index] = {column: {'details': []}}
            elif column not in self.errors[index]:
                self.errors[index][column] = {'details': []}
            self.errors[index][column]['details'].append(dict(kwargs.get('details', {}) or kwargs))
        self.masks[column] = self.masks[column] | boolmask.astype(bool)


class BaseRule(Rule):
    """
    Base rule class for Pandas DataFrame validation

    :param func: The function to apply to the DataFrame
    :param kwargs: Additional keyword arguments to pass to the function
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        """
        Verify the DataFrame against the rule
        """

        def inspect_params(f):
            signature = inspect.signature(f)
            params = {}
            if 'df' in signature.parameters:
                params['df'] = df
            if set(signature.parameters) & {'col', 'column'}:
                params['col' if 'col' in signature.parameters else 'column'] = column
            if 'error_state' in signature.parameters:
                params['error_state'] = error_state
            return params

        func_params = inspect_params(self.func)
        if len(func_params) == 3:
            return self.func(**func_params)

        if '__pre_func__' in self.kwargs:
            pre_func = self.kwargs['__pre_func__']
            df = pre_func(**inspect_params(pre_func))

        boolmask = self.func(**inspect_params(self.func))
        error_state.add_errors(boolmask, column, details=self.kwargs.get('details', {}) or self.kwargs)
        return df[column]


def rule(func, **kwargs) -> BaseRule:
    return BaseRule(func, **kwargs)


def required() -> Rule:
    @Rule.new
    def verify(df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        if column not in df.columns:
            error_state.add_errors(pd.Series(True, index=df.index), column, details=errors.MISSING)
            return pd.Series(dtype='object', index=df.index)

        error_state.add_errors(df[column].isna(), column, details=errors.MISSING)
        return df[column]

    return verify


def equal_to(eq: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] != eq,
        details=errors.EQUAL_TO.format(eq=eq)
    )


def not_equal_to(ne: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] == ne,
        details=errors.NOT_EQUAL_TO.format(ne=ne)
    )


def greater_than(gt: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] <= gt,
        details=errors.GREATER_THAN.format(gt=gt)
    )


def greater_than_equal(ge: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] < ge,
        details=errors.GREATER_THAN_EQUAL.format(ge=ge)
    )


def less_than(lt: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] >= lt,
        details=errors.LESS_THAN.format(lt=lt)
    )


def less_than_equal(le: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] > le,
        details=errors.LESS_THAN_EQUAL.format(le=le)
    )


def multiple_of(multiple: Union[int, float]) -> Rule:
    def before(df, col):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError('multiple_of rule can only be applied to numeric columns')
        return df

    if multiple <= 0:
        raise ValueError('multiple_of must be greater than zero')

    return rule(
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        lambda df, col: (df[col][df[col].notna()] < 0) | (df[col][df[col].notna()] % multiple != 0.0),
        details=errors.MULTIPLE_OF.format(multiple_of=multiple),
        __pre_func__=before
    )


def min_length(value: int) -> Rule:
    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('min_length rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: df[col][df[col].notna()].str.len() < value,
        details=errors.STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before
    )


def max_length(value: int) -> Rule:
    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('max_length rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: df[col][df[col].notna()].str.len() > value,
        details=errors.STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before
    )


def pattern(regex: str) -> Rule:
    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('pattern rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: ~df[col][df[col].notna()].str.match(regex, na=False),
        details=errors.STRING_PATTERN_MISMATCH.format(pattern=regex),
        __pre_func__=before
    )


def isin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda df, col: ~df[col][df[col].notna()].isin(values),
        details=errors.ENUM.format(expected=values)
    )


def notin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()].isin(values),
        details=errors.NOT_ENUM.format(unexpected=values)
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


def time_parsing() -> Rule:
    return RegexStringDerivedTypeRule(
        r'^(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{1,6})?(Z|[+-](2[0-3]|[01][0-9]):([0-5][0-9]))?$',
        errors.TIME_TYPE, errors.TIME_PARSING
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


def uuid_parsing() -> Rule:
    return UUIDTypeRule()


def ipv4_address() -> Rule:
    return RegexStringDerivedTypeRule(
        r'^((25[0-5]|2[0-4]\d|(1\d{2}|[1-9]\d|\d))\.){3}(25[0-5]|2[0-4]\d|(1\d{2}|[1-9]\d|\d))$',
        errors.IPV4, errors.IPV4
    )


def ipv6_address() -> Rule:
    return RegexStringDerivedTypeRule(
        r'^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:'
        r'[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0'
        r'-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}'
        r'(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7'
        r'}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4'
        r']|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1'
        r',4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$',
        errors.IPV6, errors.IPV6
    )


def uri_parsing() -> Rule:
    return RegexStringDerivedTypeRule(
        r"^([a-z][a-z0-9+.-]+):(\/\/([^@]+@)?([a-z0-9.\-_~]+)(:\d+)?)?((?:[a-z0-9-._~]|%[a-f0-9]|[!$&'"
        r"()*+,;=:@])+(?:\/(?:[a-z0-9-._~]|%[a-f0-9]|[!$&'()*+,;=:@])*)*|(?:\/(?:[a-z0-9-._~]|%[a-f0-9"
        r"]|[!$&'()*+,;=:@])+)*)?(\?(?:[a-z0-9-._~]|%[a-f0-9]|[!$&'()*+,;=:@]|[/?])+)?(\#(?:[a-z0-9-._"
        r"~]|%[a-f0-9]|[!$&'()*+,;=:@]|[/?])+)?$",
        errors.URI_TYPE, errors.URI_PARSING
    )


def object_parsing(schema) -> Rule:
    raise NotImplementedError('object_parsing is not implemented yet')


def array_parsing(schema) -> Rule:
    raise NotImplementedError('array_parsing is not implemented yet')


class NumberTypeRule(Rule):
    """
    Class for numeric type casting rules
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


class StringTypeRule(Rule):
    """
    Class for string type casting rules
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


class BooleanTypeRule(Rule):
    """
    Class for boolean type casting rules
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


class DatetimeTypeRule(Rule):
    """
    Class for datetime type casting rules
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


class StringDerivedTypeRule(Rule, abc.ABC):
    """
    Base class for rules that derive from string types
    """

    @staticmethod
    def cast_as_str(df: pd.DataFrame, column: str, error_state: ErrorState, supported, error_details) -> pd.Series:
        if df[column].dtype == 'string[python]':
            return df[column]

        if not pd.api.types.is_object_dtype(df[column]):
            error_state.add_errors(pd.Series(True, index=df.index), column, details=error_details)
            return pd.Series(dtype='string', name=column, index=df.index)

        non_parsing_type_series = df[column].apply(lambda x: not isinstance(x, supported))
        error_state.add_errors(non_parsing_type_series & df[column].notna(), column, details=error_details)

        str_series = df[column].astype('string')
        str_series[non_parsing_type_series] = None
        return str_series


class RegexStringDerivedTypeRule(StringDerivedTypeRule):
    """
    Base class for rules that derive from string types using regex matching
    """

    def __init__(self, regex: str, type_error, parsing_error):
        self.regex = regex
        self.type_error = type_error
        self.parsing_error = parsing_error

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        series = self.cast_as_str(df, column, error_state, str, self.type_error)
        boolmask = ~series.str.match(self.regex, na=False)
        error_state.add_errors(boolmask & series.notna(), column, details=self.parsing_error)
        series[boolmask] = None
        return series


class UUIDTypeRule(StringDerivedTypeRule):
    """
    Class for UUID type casting rules
    """

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        uuid_regex = '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        series = self.cast_as_str(df, column, error_state, (str, UUID), errors.UUID_TYPE)
        series = series.str.lower()
        boolmask = ~series.str.match(uuid_regex, na=False)
        error_state.add_errors(boolmask & series.notna(), column, details=errors.UUID_PARSING)
        series[boolmask] = None
        return series
