import inspect  # noqa: I001
import json
import operator
from functools import reduce
from typing import Any, Iterable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import (
    ArrayType, BooleanType, DataType, DateType, FloatType, IntegerType, MapType, NullType, NumericType, StringType,
    StructField, StructType, TimestampType,
)

from norma import errors
from norma.engines.pyspark.utils import dtype_drop, dtype_set, nested_drop_expr, nested_get_expr, nested_set_expr
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    Error state for PySpark DataFrame validation.
    """

    def __init__(self, error_column: str, schema):
        self.error_column = error_column
        self.has_array = self._has_array_column(schema)
        self.names = {}
        self.reserved_names = set()

        self.exprs = self.dtypes = self.input_columns = self.pending_errors = self.pending_indexes = \
            self.pending_backups = self.initialized_errors = self.created_roots = self.dropped_roots = None

    def add_errors(self, boolmask: Column, column: str, details=None, **kwargs) -> None:
        """
        Record an error detail expression for a column (True values in the boolmask indicate errors).
        """

        details = dict(details or kwargs.get('details') or {})
        details_lit = [fn.lit(v).alias(k) for k, v in details.items()]

        if '[]' in column:
            indexes = fn.filter(fn.transform(boolmask, fn.when), lambda x: x.isNotNull())
            details_lit.append(indexes.alias('loc'))
            details_col = fn.when(fn.array_size(indexes) > 0, fn.struct(*details_lit))

            prev = self.pending_indexes.get(column)
            self.pending_indexes[column] = boolmask if prev is None else fn.zip_with(prev, boolmask, operator.or_)
        else:
            # if DataFrame has at least one array column, we need to add indexes
            # because we cannot append a struct to an array with different types
            if self.has_array:
                details_lit.append(fn.lit(None).cast('array<int>').alias('loc'))
            details_col = fn.when(boolmask, fn.struct(*details_lit))

        self.pending_errors.setdefault(column, []).append(details_col)

    def seed(self) -> None:
        """
        Reset the symbolic accumulation state to reflect the given DataFrame.
        """

        self.exprs = {}
        self.pending_errors = {}
        self.pending_indexes = {}
        self.pending_backups = {}
        self.initialized_errors = set()
        self.created_roots = []
        self.dropped_roots = set()

    def resync(self, df: DataFrame) -> None:
        """
        Refresh the schema knowledge after a rule transformed the DataFrame directly.
        """

        self.dtypes = {field.name: field.dataType for field in df.schema.fields}
        self.input_columns = df.columns

    def flush(self, df: DataFrame) -> DataFrame:  # noqa: PLR0912
        """
        Materialize all pending symbolic state into the DataFrame using a single projection.
        """

        if not any((self.exprs, self.pending_errors, self.pending_indexes,
                    self.pending_backups, self.created_roots, self.dropped_roots)):
            return df

        existing = set(df.columns)
        selection = {}
        for column in df.columns:
            if column in self.dropped_roots:
                continue
            selection[column] = self.exprs.get(column, fn.col(column))
        for root in self.created_roots:
            selection[root] = self.exprs[root]

        for path, pending in self.pending_errors.items():
            name = self.errors_col(path)
            if path in self.initialized_errors:
                arr = fn.array(*pending).cast(self._details_dtype()) if pending else self._empty_errors_details()
            else:
                arr = fn.array(*pending)
            if name in existing:
                selection[name] = fn.concat(fn.col(name), arr) if pending else fn.col(name)
            else:
                selection[name] = arr

        for path, indexes in self.pending_indexes.items():
            name = self.indexes_col(path)
            if name in existing:
                selection[name] = fn.zip_with(fn.col(name), indexes, operator.or_)
            else:
                selection[name] = indexes

        for path, backup in self.pending_backups.items():
            selection[self.backup_col(path)] = backup

        result = df.select(*(expr.alias(name) for name, expr in selection.items()))
        self.seed()
        self.resync(result)

        return result

    def expr_of(self, column: str) -> Column:
        """
        Return the current value expression of a column (flattened values for array columns).
        """

        return nested_get_expr(column, self.exprs.get(self._root_of(column)))

    def dtype_of(self, column: str) -> DataType:
        """
        Return the current data type of column, mirroring dtype_at on the symbolic state.
        """

        def data_type(dtype, col):
            if not isinstance(dtype, StructType):
                raise ValueError(f'Column "{column}" is not a nested column in the DataFrame schema.')
            if '[]' in col:
                return dtype[col[:-2]].dataType.elementType
            return dtype[col].dataType

        parts = column.split('.')
        first = parts[0]
        dtype = self.dtypes[first.removesuffix('[]')]
        if first.endswith('[]'):
            dtype = dtype.elementType
        for part in parts[1:]:
            dtype = data_type(dtype, part)
        return dtype

    def set_expr(self, column: str, val) -> None:
        """
        Set the value expression of a (possibly nested) column, rebuilding the root expression.
        """

        root = self._root_of(column)
        self.exprs[root] = nested_set_expr(column, val, self.exprs.get(root), self.dtypes.get(root))
        if root not in self.input_columns and root not in self.created_roots:
            self.created_roots.append(root)

    def set_dtype(self, column: str, dtype: DataType) -> None:
        """
        Update the tracked data type of (possibly nested) column.
        """

        parts = column.split('.')
        first = parts[0]
        root = first.removesuffix('[]')
        rest = '.'.join(parts[1:])

        if first.endswith('[]'):
            base = self.dtypes.get(root)
            elem = base.elementType if isinstance(base, ArrayType) else None
            self.dtypes[root] = ArrayType(dtype_set(elem, rest, dtype) if rest else dtype)
        elif rest:
            self.dtypes[root] = dtype_set(self.dtypes.get(root), rest, dtype)
        else:
            self.dtypes[root] = dtype

    def drop_column(self, column: str) -> None:
        """
        Drop a (possibly nested) column from the symbolic state.
        """

        parts = column.split('.')
        first = parts[0]
        root = first.removesuffix('[]')
        rest = '.'.join(parts[1:])

        if not rest:
            self.dropped_roots.add(root)
            self.exprs.pop(root, None)
            self.dtypes.pop(root, None)
            return

        self.exprs[root] = nested_drop_expr(column, self.exprs.get(root, fn.col(root)), self.dtypes.get(root))
        if first.endswith('[]'):
            base = self.dtypes[root]
            self.dtypes[root] = ArrayType(dtype_drop(base.elementType, rest), base.containsNull)
        else:
            self.dtypes[root] = dtype_drop(self.dtypes[root], rest)

    def set_backup(self, column: str) -> None:
        """
        Snapshot the current value expression of a column as its backup.
        """

        self.pending_backups[column] = self.expr_of(column)

    def backup_expr(self, column: str) -> Column:
        """
        Return the backup expression of a column (or a reference to the materialized backup column).
        """

        if column in self.pending_backups:
            return self.pending_backups[column]
        return fn.col(self.backup_col(column))

    def track_column(self, column: str) -> None:
        """
        Start tracking a column: ensure it exists in the symbolic state (creating a void
        column if missing) and force its error-details column to be materialized on flush.
        """

        try:
            self.dtype_of(column)
        except KeyError:
            self.set_expr(column, fn.lit(None).cast('void'))
            self.set_dtype(column, NullType())

        self.pending_errors.setdefault(column, [])
        self.initialized_errors.add(column)

    def errors_col(self, column: str) -> str:
        """
        Return the name of the error-details column for a given column.
        """

        return f'{self.error_column}_{self._name_of(column)}'

    def indexes_col(self, column: str) -> str:
        """
        Return the name of the invalid-indexes column for a given array column.
        """

        return f'{self.error_column}_{self._name_of(column)}_idx'

    def backup_col(self, column: str) -> str:
        """
        Return the name of the backup column for a given column.
        """

        return f'{self.error_column}_{self._name_of(column)}_bak'

    def _name_of(self, column: str) -> str:
        """
        Return a deterministic per-column base for service column names, avoiding
        collisions with input columns and with bases already handed out.
        """

        if column in self.names:
            return self.names[column]

        base = column.replace('[]', '_arr').replace('.', '_')
        name, attempt = base, 1
        while not self._reserve(name):
            attempt += 1
            name = f'{base}_{attempt}'
        self.names[column] = name
        return name

    def _reserve(self, name: str) -> bool:
        derived = {f'{self.error_column}_{name}{kind}' for kind in ('', '_idx', '_bak')}
        if derived & self.reserved_names or derived & set(self.input_columns or ()):
            return False
        self.reserved_names |= derived
        return True

    def _details_dtype(self):
        loc = ''
        if self.has_array:
            loc = ',loc:array<int>'
        return f'array<struct<type:string,msg:string{loc}>>'

    def _empty_errors_details(self):
        return fn.array().cast(self._details_dtype())

    @staticmethod
    def _root_of(column):
        root = column.split('.')[0]
        return root.removesuffix('[]')

    @staticmethod
    def _has_array_column(schema):
        has_array = False
        for name in schema.nested_columns:
            if '[]' in name:
                has_array = True
            if name.count('[]') > 1:
                raise NotImplementedError('nested arrays are not supported yet')
        return has_array


class BaseRule(Rule):
    """
    Base rule class for PySpark DataFrame validation

    :param func: The function to apply to the DataFrame
    :param kwargs: Additional keyword arguments to pass to the function
    """

    accumulates = True

    def __init__(self, func, details=None, **kwargs):
        self.func = func
        self.details = details
        self.kwargs = kwargs

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        """
        Verify the DataFrame against the rule
        """

        def inspect_params(f, col_expr):
            signature = inspect.signature(f)
            params = {}
            if 'df' in signature.parameters:
                params['df'] = df
            if set(signature.parameters) & {'col', 'column'}:
                params['col' if 'col' in signature.parameters else 'column'] = column
            if set(signature.parameters) & {'col_expr', 'column_expr'}:
                params['col_expr' if 'col_expr' in signature.parameters else 'column_expr'] = col_expr
            if 'error_state' in signature.parameters:
                params['error_state'] = error_state
            return params

        signature = inspect.signature(self.func)
        if 'df' in signature.parameters:
            # the rule transforms the DataFrame directly: materialize the pending state first
            df = error_state.flush(df)
            df = self.func(**inspect_params(self.func, fn.col(column)))
            error_state.resync(df)
            return df

        if '__pre_func__' in self.kwargs:
            pre_func = self.kwargs['__pre_func__']
            pre_func(**inspect_params(pre_func, None))

        if '[]' not in column:
            if set(signature.parameters) & {'col', 'column'}:
                # the rule builds fn.col(name) itself: the name must resolve to the current value
                df = error_state.flush(df)
            boolmask = self.func(**inspect_params(self.func, error_state.expr_of(column)))
            error_state.add_errors(boolmask, column, self.details)
            return df

        indexes = fn.transform(error_state.expr_of(column), self.func)
        error_state.add_errors(indexes, column, self.details)
        return df


def rule(func, **kwargs) -> BaseRule:
    return BaseRule(func, **kwargs)


def required() -> Rule:
    return rule(
        lambda col_expr: fn.isnull(col_expr),  # noqa: PLW0108
        details=errors.MISSING,
    )


def equal_to(eq: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr != fn.lit(eq),
        details=errors.EQUAL_TO.format(eq=eq),
    )


def not_equal_to(ne: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr == fn.lit(ne),
        details=errors.NOT_EQUAL_TO.format(ne=ne),
    )


def greater_than(gt: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr <= fn.lit(gt),
        details=errors.GREATER_THAN.format(gt=gt),
    )


def greater_than_equal(ge: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr < fn.lit(ge),
        details=errors.GREATER_THAN_EQUAL.format(ge=ge),
    )


def less_than(lt: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr >= fn.lit(lt),
        details=errors.LESS_THAN.format(lt=lt),
    )


def less_than_equal(le: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr > fn.lit(le),
        details=errors.LESS_THAN_EQUAL.format(le=le),
    )


def multiple_of(multiple: float) -> Rule:
    def before(col, error_state):
        data_type = error_state.dtype_of(col)
        if not isinstance(data_type, NumericType):
            raise ValueError('multiple_of rule can only be applied to numeric columns')

    if multiple <= 0:
        raise ValueError('multiple_of must be greater than zero')

    return rule(
        lambda col_expr: (col_expr < fn.lit(0)) | ((col_expr % fn.lit(multiple)) != fn.lit(0)),
        details=errors.MULTIPLE_OF.format(multiple_of=multiple),
        __pre_func__=before,
    )


def min_length(value: int) -> Rule:
    def before(col, error_state):
        data_type = error_state.dtype_of(col)
        if not isinstance(data_type, StringType):
            raise ValueError('min_length rule can only be applied to string columns')

    return rule(
        lambda col_expr: fn.length(col_expr) < value,
        details=errors.STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before,
    )


def max_length(value: int) -> Rule:
    def before(col, error_state):
        data_type = error_state.dtype_of(col)
        if not isinstance(data_type, StringType):
            raise ValueError('max_length rule can only be applied to string columns')

    return rule(
        lambda col_expr: fn.length(col_expr) > value,
        details=errors.STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before,
    )


def pattern(regex: str) -> Rule:
    def before(col, error_state):
        data_type = error_state.dtype_of(col)
        if not isinstance(data_type, StringType):
            raise ValueError('pattern rule can only be applied to string columns')

    return rule(
        lambda col_expr: ~col_expr.rlike(regex),
        details=errors.STRING_PATTERN_MISMATCH.format(pattern=regex),
        __pre_func__=before,
    )


def isin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda col_expr: ~col_expr.isin(values),
        details=errors.ENUM.format(expected=values),
    )


def notin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda col_expr: col_expr.isin(values),
        details=errors.NOT_ENUM.format(unexpected=values),
    )


def unique_items() -> Rule:
    def before(col, error_state):
        data_type = error_state.dtype_of(col)
        if not isinstance(data_type, ArrayType):
            raise ValueError('unique_items rule can only be applied to array columns')

    return rule(
        lambda col_expr: fn.size(col_expr) != fn.size(fn.array_distinct(col_expr)),
        details=errors.UNIQUE_ITEMS,
        __pre_func__=before,
    )


def max_items(value: int) -> Rule:
    def before(col, error_state):
        data_type = error_state.dtype_of(col)
        if not isinstance(data_type, ArrayType):
            raise ValueError('max_items rule can only be applied to array columns')

    return rule(
        lambda col_expr: fn.array_size(col_expr) > value,
        details=errors.TOO_LONG.format(_type_='Array', max_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before,
    )


def min_items(value: int) -> Rule:
    def before(col, error_state):
        data_type = error_state.dtype_of(col)
        if not isinstance(data_type, ArrayType):
            raise ValueError('min_items rule can only be applied to array columns')

    return rule(
        lambda col_expr: fn.array_size(col_expr) < value,
        details=errors.TOO_SHORT.format(_type_='Array', min_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before,
    )


class ExtraForbiddenRule(Rule):
    """
    Rule that forbids columns not defined in the schema, moving their values to backup columns
    """

    accumulates = True

    def __init__(self, allowed: Iterable[str]):
        self.allowed = allowed

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        if column in self.allowed:
            return df

        error_state.set_backup(column)
        if '[]' in column and not column.endswith('[]'):
            error_state.drop_column(column)
            error_state.add_errors(
                fn.transform(error_state.expr_of(column.split('[]', maxsplit=1)[0]), lambda _: fn.lit(True)),
                column,
                errors.EXTRA_FORBIDDEN,
            )
            return df

        error_state.drop_column(column.removesuffix('[]'))
        error_state.add_errors(fn.lit(True), column, errors.EXTRA_FORBIDDEN)
        return df


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    return ExtraForbiddenRule(allowed)


def int_parsing() -> Rule:
    return DataTypeRule(
        lambda col: col.cast('integer'),
        IntegerType,
        (StringType, NumericType, BooleanType),
        errors.INT_TYPE,
        errors.INT_PARSING,
    )


def float_parsing():
    return DataTypeRule(
        lambda col: col.cast('float'),
        FloatType,
        (StringType, NumericType, BooleanType),
        errors.FLOAT_TYPE,
        errors.FLOAT_PARSING,
    )


def str_parsing() -> Rule:
    return DataTypeRule(
        lambda col: col.cast('string'),
        StringType,
        (NumericType, BooleanType, DateType, TimestampType),
        errors.STRING_TYPE,
    )


def bool_parsing() -> Rule:
    return BooleanTypeRule(
        lambda col: col.cast('boolean'),
        BooleanType,
        (NumericType, StringType),
        errors.BOOL_TYPE,
        errors.BOOL_PARSING,
    )


def datetime_parsing() -> Rule:
    return DataTypeRule(
        lambda col: fn.to_timestamp(col),  # noqa: PLW0108
        TimestampType,
        (StringType, DateType),
        errors.DATETIME_TYPE,
        errors.DATETIME_PARSING,
    )


def date_parsing() -> Rule:
    return DataTypeRule(
        lambda col: fn.to_date(fn.to_timestamp(col)),
        DateType,
        (StringType, TimestampType),
        errors.DATE_TYPE,
        errors.DATE_PARSING,
    )


def time_parsing() -> Rule:
    time_regex = (
        r'^(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{1,6})?(Z|[+-](2[0-3]|[01][0-9]):([0-5][0-9]))?$'
    )
    return ComplexTypeRule(
        lambda col: fn.when(col.rlike(time_regex), col),
        StringType,
        (StringType,),
        errors.TIME_TYPE,
        errors.TIME_PARSING,
    )


def duration_parsing() -> Rule:
    duration_regex = r'^-?P(?=\d|T\d)(\d+Y)?(\d+M)?(\d+D)?(T(?=\d)(\d+H)?(\d+M)?(\d+(\.\d+)?S)?)?$'
    return ComplexTypeRule(
        lambda col: fn.when(col.rlike(duration_regex), col),
        StringType,
        (StringType,),
        errors.DURATION_TYPE,
        errors.DURATION_PARSING,
    )


def uuid_parsing() -> Rule:
    uuid_regex = '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    return ComplexTypeRule(
        lambda col: fn.when(fn.lower(col).rlike(uuid_regex), fn.lower(col)),
        StringType,
        (StringType,),
        errors.UUID_TYPE,
        errors.UUID_PARSING,
    )


def ipv4_address() -> Rule:
    ipv4_regex = r'^((25[0-5]|2[0-4]\d|(1\d{2}|[1-9]\d|\d))\.){3}(25[0-5]|2[0-4]\d|(1\d{2}|[1-9]\d|\d))$'
    return ComplexTypeRule(
        lambda col: fn.when(col.rlike(ipv4_regex), col),
        StringType,
        (StringType,),
        errors.IPV4,
        errors.IPV4,
    )


def ipv6_address() -> Rule:
    # Regular expression to match valid IPv6 addresses
    # link: https://stackoverflow.com/questions/53497/regular-expression-that-matches-valid-ipv6-addresses
    ipv6_regex = (
        r'^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:'
        r'[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0'
        r'-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}'
        r'(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7'
        r'}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4'
        r']|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1'
        r',4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$'
    )
    return ComplexTypeRule(
        lambda col: fn.when(col.rlike(ipv6_regex), col),
        StringType,
        (StringType,),
        errors.IPV6,
        errors.IPV6,
    )


def uri_parsing() -> Rule:
    uri_regex = (
        r"^([a-z][a-z0-9+.-]+):(\/\/([^@]+@)?([a-z0-9.\-_~]+)(:\d+)?)?((?:[a-z0-9-._~]|%[a-f0-9]|[!$&'"
        r"()*+,;=:@])+(?:\/(?:[a-z0-9-._~]|%[a-f0-9]|[!$&'()*+,;=:@])*)*|(?:\/(?:[a-z0-9-._~]|%[a-f0-9"
        r"]|[!$&'()*+,;=:@])+)*)?(\?(?:[a-z0-9-._~]|%[a-f0-9]|[!$&'()*+,;=:@]|[/?])+)?(\#(?:[a-z0-9-._"
        r"~]|%[a-f0-9]|[!$&'()*+,;=:@]|[/?])+)?$"
    )

    return ComplexTypeRule(
        lambda col: fn.when(col.rlike(uri_regex), col),
        StringType,
        (StringType,),
        errors.URI_TYPE,
        errors.URI_PARSING,
    )


def object_parsing(schema) -> Rule:
    return ObjectTypeRule(schema)


def array_parsing(schema) -> Rule:
    return ArrayTypeRule(schema)


class DataTypeRule(Rule):
    """
    Abstract base class for casting to target data type
    """

    accumulates = True

    def __init__(self, caster, dtype, supported_cast_dtypes, type_error_details=None, parsing_error_details=None):
        self.caster = caster
        self.dtype = dtype
        self.supported_cast_dtypes = supported_cast_dtypes
        self.type_error_details = type_error_details
        self.parsing_error_details = parsing_error_details

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        data_type = error_state.dtype_of(column)
        if self._is_valid_dtype(data_type):
            return df

        if data_type.typeName() == 'void':
            error_state.set_expr(column, fn.lit(None).cast(self._target_dtype()))
            error_state.set_dtype(column, self._target_dtype())
            return df

        if '[]' in column:
            return self._verify_array(df, column, data_type, error_state)
        return self._verify_scalar(df, column, data_type, error_state)

    def _verify_scalar(self, df: DataFrame, column: str, data_type: DataType, error_state: ErrorState) -> DataFrame:
        error_state.set_backup(column)
        if not isinstance(data_type, self.supported_cast_dtypes):
            error_state.set_expr(column, fn.lit(None).cast(self._target_dtype()))
            error_state.set_dtype(column, self._target_dtype())
            error_state.add_errors(fn.lit(True), column, self.type_error_details)
            return df

        self._cast_scalar(column, data_type, error_state)
        error_state.set_dtype(column, self._cast_result_dtype(data_type))
        if not self.parsing_error_details:
            return df

        error_state.add_errors(
            fn.isnull(error_state.expr_of(column)) & fn.isnotnull(error_state.backup_expr(column)),
            column,
            self.parsing_error_details,
        )
        return df

    def _verify_array(self, df: DataFrame, column: str, element_type: DataType, error_state: ErrorState) -> DataFrame:
        error_state.set_backup(column)
        if not isinstance(element_type, self.supported_cast_dtypes):
            indexes = fn.transform(error_state.expr_of(column.split('[]', maxsplit=1)[0]), lambda _: fn.lit(True))
            error_state.set_expr(column, fn.lit(None).cast(self._target_dtype()))
            error_state.set_dtype(column, self._target_dtype())
            error_state.add_errors(indexes, column, self.type_error_details)
            return df

        self._cast_array(column, element_type, error_state)
        error_state.set_dtype(column, self._cast_result_dtype(element_type))
        if not self.parsing_error_details:
            return df

        actual_values = error_state.expr_of(column)
        indexes = fn.zip_with(
            actual_values, error_state.backup_expr(column), lambda x, y: fn.isnull(x) & fn.isnotnull(y)
        )
        error_state.add_errors(indexes, column, self.parsing_error_details)
        return df

    def _is_valid_dtype(self, data_type) -> bool:
        return isinstance(data_type, self.dtype)

    def _target_dtype(self):
        return self.dtype()

    def _cast_result_dtype(self, data_type):  # noqa: ARG002
        return self._target_dtype()

    def _cast_scalar(self, column, data_type, error_state):  # noqa: ARG002
        error_state.set_expr(column, self.caster(error_state.expr_of(column)))

    def _cast_array(self, column, element_type, error_state):  # noqa: ARG002
        error_state.set_expr(column, self.caster)


class ComplexTypeRule(DataTypeRule):
    """
    Class for casting to complex string types
    """

    def _is_valid_dtype(self, data_type) -> bool:  # noqa: ARG002
        return False


class BooleanTypeRule(DataTypeRule):
    """
    Class for casting to boolean type
    """

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        # special case for string columns where we want to cast 'on'/'off' to boolean
        def cast_str_as_bool(col):
            expr = fn.lower(fn.trim(col))
            expr = fn.when(expr.isin(['on', 'off']), expr == 'on').otherwise(col.cast('boolean'))
            return expr.cast('boolean')

        data_type = error_state.dtype_of(column)
        if isinstance(data_type, StringType) or (
            isinstance(data_type, ArrayType) and isinstance(data_type.elementType, StringType)
        ):
            self.caster = cast_str_as_bool

        return super().verify(df, column, error_state)


class ObjectTypeRule(DataTypeRule):
    """
    Class for casting to object type
    """

    def __init__(self, schema):
        super().__init__(None, StructType, (StringType, MapType), errors.OBJECT_TYPE, errors.OBJECT_PARSING)
        self.struct_type = self.parse_struct_type(schema)

    def _is_valid_dtype(self, data_type) -> bool:
        return isinstance(data_type, StructType)

    def _target_dtype(self):
        return self.struct_type

    def _cast_result_dtype(self, data_type):
        # a map cast keeps the map's value type in every struct field
        if isinstance(data_type, MapType):
            return StructType(
                [
                    StructField(name, data_type.valueType, nullable=True, metadata={})
                    for name in self.struct_type.fieldNames()
                ]
            )
        return self._target_dtype()

    def _cast_scalar(self, column, data_type, error_state):
        if isinstance(data_type, MapType):
            col_expr = error_state.expr_of(column)
            new_struct = fn.struct(*(col_expr[field].alias(field) for field in self.struct_type.fieldNames()))
            error_state.set_expr(column, new_struct)
            return

        # Workaround for Spark issue where from_json does not return null for malformed JSON
        # As result we need to check if all fields are null and the original value is not null
        # then for those cases we should try to parse the JSON using Python json.loads
        # and if it fails then we consider it as malformed JSON
        @fn.udf(returnType=BooleanType())
        def malformed_json_udf(struct_fields_is_null_and_origin_is_not_null, val):
            if not struct_fields_is_null_and_origin_is_not_null:
                return False

            try:
                json.loads(val)
                return False
            except:  # noqa: E722
                return True

        error_state.set_expr(column, fn.from_json(error_state.expr_of(column), self.struct_type))
        parsed = error_state.expr_of(column)
        backup = error_state.backup_expr(column)
        is_malformed = malformed_json_udf(
            reduce(operator.and_, (fn.isnull(parsed.getField(field)) for field in self.struct_type.fieldNames()))
            & fn.isnotnull(backup),
            backup,
        )

        error_state.set_expr(column, fn.when(~is_malformed, error_state.expr_of(column)))

    def _cast_array(self, column, element_type, error_state):
        if isinstance(element_type, MapType):

            def new_struct(x):
                return fn.struct(*(x[field].alias(field) for field in self.struct_type.fieldNames()))

            error_state.set_expr(column, new_struct)
            return

        # TODO: malformed JSON detection for nested array elements
        # for some reason to use same workaround as in _cast_scalar does not work in nested arrays
        # so currently we do not support malformed JSON detection in nested arrays
        # this should be revisited in future
        error_state.set_expr(column, lambda x: fn.from_json(x, self.struct_type))

    @staticmethod
    def parse_struct_type(schema) -> StructType:
        return StructType([StructField(name, StringType(), nullable=True, metadata={}) for name in schema.columns])


class ArrayTypeRule(DataTypeRule):
    """
    Class for casting to array type
    """

    def __init__(self, schema):
        super().__init__(None, ArrayType, (StringType, ArrayType), errors.ARRAY_TYPE, errors.ARRAY_PARSING)
        self.struct_type = self.parse_array_type(schema)

    def _verify_array(self, df: DataFrame, column: str, element_type: DataType, error_state: ErrorState) -> DataFrame:
        raise NotImplementedError('nested arrays are not supported yet')

    def _target_dtype(self):
        return self.struct_type

    def _cast_scalar(self, column, data_type, error_state):  # noqa: ARG002
        # TODO: malformed JSON detection for nested array elements
        # same issue as in ObjectTypeRule._cast_scalar with from_json not returning null for malformed JSON
        error_state.set_expr(column, fn.from_json(error_state.expr_of(column), self.struct_type))

    @staticmethod
    def parse_array_type(schema) -> ArrayType:
        if schema.inner_schema is None:
            return ArrayType(StringType())
        if schema.dtype in {'array', 'list'}:
            raise NotImplementedError('nested arrays are not supported yet')
        return ArrayType(ObjectTypeRule.parse_struct_type(schema.inner_schema))
