import abc
import inspect
import json
from collections import defaultdict
from itertools import starmap
from typing import Any, Iterable, Optional
from uuid import UUID

import pandas as pd

from norma import errors
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    Error state for Pandas DataFrame validation

    :param index: The index of the original DataFrame
    :param has_array: Whether the schema contains at least one array column
    """

    def __init__(self, index: pd.Index, has_array: bool = False) -> None:
        self.masks = defaultdict(lambda: pd.Series(False, index=index))
        self.errors = {}
        self.backups = {}
        self.has_array = has_array

    def add_errors(self, boolmask, column, **kwargs):
        """
        Add errors to the error state for a given column
        """

        details = dict(kwargs.get('details', {}) or kwargs)

        if isinstance(boolmask.index, pd.MultiIndex):
            self._add_element_errors(boolmask, column, details)
            return

        if self.has_array:
            details['loc'] = None

        for index in boolmask[boolmask].index:
            if index not in self.errors:
                self.errors[index] = {column: {'details': []}}
            elif column not in self.errors[index]:
                self.errors[index][column] = {'details': []}
            self.errors[index][column]['details'].append(dict(details))
        self.masks[column] = self.masks[column] | boolmask.astype(bool)

    def _add_element_errors(self, boolmask, column, details):
        """
        Add per-element errors for an array column (boolmask is indexed by (row, position))
        """

        flagged = boolmask[boolmask.astype(bool)]
        for index in flagged.index.get_level_values(0).unique():
            positions = sorted(int(position) for position in flagged.loc[index].index)
            if index not in self.errors:
                self.errors[index] = {column: {'details': []}}
            elif column not in self.errors[index]:
                self.errors[index][column] = {'details': []}
            self.errors[index][column]['details'].append({**details, 'loc': positions})

        if column in self.masks:
            mask = self.masks[column]
            index = mask.index.union(boolmask.index)
            self.masks[column] = \
                mask.reindex(index, fill_value=False) | boolmask.reindex(index, fill_value=False).astype(bool)
        else:
            self.masks[column] = boolmask.astype(bool)

    def set_backup(self, column, series):
        """
        Snapshot the pre-cast values of a column for error reporting (first backup wins)
        """

        if column not in self.backups:
            self.backups[column] = series.copy()


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
        details=errors.EQUAL_TO.format(eq=eq),
    )


def not_equal_to(ne: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] == ne,
        details=errors.NOT_EQUAL_TO.format(ne=ne),
    )


def greater_than(gt: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] <= gt,
        details=errors.GREATER_THAN.format(gt=gt),
    )


def greater_than_equal(ge: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] < ge,
        details=errors.GREATER_THAN_EQUAL.format(ge=ge),
    )


def less_than(lt: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] >= lt,
        details=errors.LESS_THAN.format(lt=lt),
    )


def less_than_equal(le: Any) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()] > le,
        details=errors.LESS_THAN_EQUAL.format(le=le),
    )


def multiple_of(multiple: float) -> Rule:
    def before(df, col):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError('multiple_of rule can only be applied to numeric columns')
        return df

    if multiple <= 0:
        raise ValueError('multiple_of must be greater than zero')

    return rule(
        lambda df, col: (df[col][df[col].notna()] < 0) | (df[col][df[col].notna()] % multiple != 0.0),  # noqa: RUF069
        details=errors.MULTIPLE_OF.format(multiple_of=multiple),
        __pre_func__=before,
    )


def min_length(value: int) -> Rule:
    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('min_length rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: df[col][df[col].notna()].str.len() < value,
        details=errors.STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before,
    )


def max_length(value: int) -> Rule:
    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('max_length rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: df[col][df[col].notna()].str.len() > value,
        details=errors.STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before,
    )


def pattern(regex: str) -> Rule:
    def before(df, column):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError('pattern rule can only be applied to string columns')
        return df

    return rule(
        lambda df, col: ~df[col][df[col].notna()].str.match(regex, na=False),
        details=errors.STRING_PATTERN_MISMATCH.format(pattern=regex),
        __pre_func__=before,
    )


def isin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda df, col: ~df[col][df[col].notna()].isin(values),
        details=errors.ENUM.format(expected=values),
    )


def notin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()].isin(values),
        details=errors.NOT_ENUM.format(unexpected=values),
    )


def unique_items() -> Rule:
    def has_duplicates(items):
        keys = [json.dumps(item, sort_keys=True, default=str) for item in items]
        return len(keys) != len(set(keys))

    return rule(
        lambda df, col: df[col][df[col].notna()].apply(has_duplicates),
        details=errors.UNIQUE_ITEMS,
        __pre_func__=_ensure_array_column('unique_items'),
    )


def max_items(value: int) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()].apply(len) > value,
        details=errors.TOO_LONG.format(_type_='Array', max_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=_ensure_array_column('max_items'),
    )


def min_items(value: int) -> Rule:
    return rule(
        lambda df, col: df[col][df[col].notna()].apply(len) < value,
        details=errors.TOO_SHORT.format(_type_='Array', min_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=_ensure_array_column('min_items'),
    )


def _ensure_array_column(rule_name):
    def before(df, column):
        series = df[column]
        is_array = pd.api.types.is_object_dtype(series) \
            and series.dropna().apply(lambda x: isinstance(x, list)).all()
        if not (is_array or series.isna().all()):
            raise ValueError(f'{rule_name} rule can only be applied to array columns')
        return df

    return before


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
        errors.TIME_TYPE,
        errors.TIME_PARSING,
    )


def duration_parsing() -> Rule:
    return RegexStringDerivedTypeRule(
        r'^-?P(?=\d|T\d)(\d+Y)?(\d+M)?(\d+D)?(T(?=\d)(\d+H)?(\d+M)?(\d+(\.\d+)?S)?)?$',
        errors.DURATION_TYPE,
        errors.DURATION_PARSING,
    )


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    @Rule.new
    def verify(df: pd.DataFrame, column: str, error_state: ErrorState) -> Optional[pd.Series]:
        if column in allowed:
            return df[column]

        error_state.set_backup(column, df[column])
        error_state.add_errors(pd.Series(True, index=df.index), column, details=errors.EXTRA_FORBIDDEN)

        error_state.masks.pop(column, None)
        df.drop(column, axis=1, inplace=True)  # noqa: PD002
        return None

    return verify


def uuid_parsing() -> Rule:
    return UUIDTypeRule()


def ipv4_address() -> Rule:
    return RegexStringDerivedTypeRule(
        r'^((25[0-5]|2[0-4]\d|(1\d{2}|[1-9]\d|\d))\.){3}(25[0-5]|2[0-4]\d|(1\d{2}|[1-9]\d|\d))$',
        errors.IPV4,
        errors.IPV4,
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
        errors.IPV6,
        errors.IPV6,
    )


def uri_parsing() -> Rule:
    return RegexStringDerivedTypeRule(
        r"^([a-z][a-z0-9+.-]+):(\/\/([^@]+@)?([a-z0-9.\-_~]+)(:\d+)?)?((?:[a-z0-9-._~]|%[a-f0-9]|[!$&'"
        r"()*+,;=:@])+(?:\/(?:[a-z0-9-._~]|%[a-f0-9]|[!$&'()*+,;=:@])*)*|(?:\/(?:[a-z0-9-._~]|%[a-f0-9"
        r"]|[!$&'()*+,;=:@])+)*)?(\?(?:[a-z0-9-._~]|%[a-f0-9]|[!$&'()*+,;=:@]|[/?])+)?(\#(?:[a-z0-9-._"
        r"~]|%[a-f0-9]|[!$&'()*+,;=:@]|[/?])+)?$",
        errors.URI_TYPE,
        errors.URI_PARSING,
    )


def object_parsing(schema) -> Rule:
    return ObjectTypeRule(schema)


def array_parsing(schema) -> Rule:
    return ArrayTypeRule(schema)


def _stringify(value):
    """
    Render a JSON-parsed value the way Spark casts it to an all-string struct field
    """

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(',', ':'))
    return json.dumps(value)


def _is_null(value):
    return value is None or (pd.api.types.is_scalar(value) and pd.isna(value))


class ObjectTypeRule(Rule):
    """
    Class for object type casting rules

    :param schema: The inner schema describing the object fields
    """

    def __init__(self, schema):
        self.schema = schema

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        error_state.set_backup(column, df[column])
        # array elements mirror Spark's from_json semantics where malformed JSON silently becomes null
        is_element = isinstance(df.index, pd.MultiIndex)

        type_mask = pd.Series(False, index=df.index)
        parsing_mask = pd.Series(False, index=df.index)

        def parse(index, value):
            if _is_null(value):
                return None
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    parsing_mask[index] = not is_element
                    return None
                if not isinstance(parsed, dict):
                    return None
                return {key: _stringify(parsed[key]) for key in self.schema.columns if key in parsed}
            type_mask[index] = True
            return None

        series = pd.Series(list(starmap(parse, df[column].items())), index=df.index, dtype='object')
        error_state.add_errors(type_mask, column, details=errors.OBJECT_TYPE)
        error_state.add_errors(parsing_mask, column, details=errors.OBJECT_PARSING)
        return series


class ArrayTypeRule(Rule):
    """
    Class for array type casting rules

    :param schema: The column describing the array elements
    """

    def __init__(self, schema):
        self.inner_column = schema

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        if '[]' in column:
            raise NotImplementedError('nested arrays are not supported yet')

        error_state.set_backup(column, df[column])
        inner_schema = self.inner_column.inner_schema if self.inner_column is not None else None

        type_mask = pd.Series(False, index=df.index)
        parsing_mask = pd.Series(False, index=df.index)

        def parse_element(element):
            if inner_schema is not None:
                if isinstance(element, dict):
                    return {key: _stringify(element[key]) for key in inner_schema.columns if key in element}
                return None
            return _stringify(element)

        def parse(index, value):
            if _is_null(value):
                return None
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    parsing_mask[index] = True
                    return None
                if isinstance(parsed, dict):
                    parsed = [parsed]
                if not isinstance(parsed, list):
                    parsing_mask[index] = True
                    return None
                return [parse_element(element) for element in parsed]
            type_mask[index] = True
            return None

        series = pd.Series(list(starmap(parse, df[column].items())), index=df.index, dtype='object')
        error_state.add_errors(type_mask, column, details=errors.ARRAY_TYPE)
        error_state.add_errors(parsing_mask, column, details=errors.ARRAY_PARSING)
        return series


class NumberTypeRule(Rule):
    """
    Class for numeric type casting rules
    """

    def __init__(self, dtype, numeric_type, numeric_parsing):
        self.dtype = dtype
        self.numeric_type = numeric_type
        self.numeric_parsing = numeric_parsing

    def verify(self, df: pd.DataFrame, column: str, error_state: ErrorState) -> pd.Series:
        error_state.set_backup(column, df[column])
        if df[column].dtype == self.dtype:
            return df[column]

        non_parsing_type_series = pd.Series(False, index=df.index)
        if not (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_numeric_dtype(df[column])):
            if not pd.api.types.is_object_dtype(df[column]):
                error_state.add_errors(pd.Series(True, index=df.index), column, details=self.numeric_type)
                return pd.Series(dtype=self.dtype, name=column, index=df.index)

            non_parsing_type_series = \
                df[column].apply(lambda x: not isinstance(x, (str, bool, int, float))) & df[column].notna()
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
        error_state.set_backup(column, df[column])
        if df[column].dtype == 'string[python]':
            return df[column]

        non_parsing_type_series = pd.Series(False, index=df.index)
        bool_series = pd.Series(False, index=df.index)
        if pd.api.types.is_object_dtype(df[column]):
            non_parsing_type_series = \
                df[column].apply(lambda x: not isinstance(x, (str, bool, int, float))) & df[column].notna()
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
        error_state.set_backup(column, df[column])
        if pd.api.types.is_bool_dtype(df[column]):
            return df[column].astype('boolean')

        non_parsing_type_series = pd.Series(False, index=df.index)
        if not (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_numeric_dtype(df[column])):
            if not pd.api.types.is_object_dtype(df[column]):
                error_state.add_errors(pd.Series(True, index=df.index), column, details=errors.BOOL_TYPE)
                return pd.Series(dtype='boolean', name=column, index=df.index)

            non_parsing_type_series = \
                df[column].apply(lambda x: not isinstance(x, (str, bool, int, float))) & df[column].notna()
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
        error_state.set_backup(column, df[column])
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
            datetime_series = pd.Series(datetime_series.values.astype(self.dtype), name=column)  # noqa: PD011

        boolmask = datetime_series.isna() & df[column].notna() & ~non_parsing_type_series
        error_state.add_errors(boolmask, column, details=self.dt_parsing)
        return datetime_series


class StringDerivedTypeRule(Rule, abc.ABC):
    """
    Base class for rules that derive from string types
    """

    @staticmethod
    def cast_as_str(df: pd.DataFrame, column: str, error_state: ErrorState, supported, error_details) -> pd.Series:
        error_state.set_backup(column, df[column])
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
