import inspect  # noqa: I001
import json
import operator
from typing import Any, Callable, Iterable, Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import (
    ArrayType, BooleanType, DataType, DateType, FloatType, IntegerType, MapType, NullType, NumericType, StringType,
    StructField, StructType, TimestampType,
)

from norma import errors
from norma.engines.pyspark.utils import (
    StructOverlay, dtype_at, dtype_drop, dtype_set, format_value, nested_drop_expr, nested_get_expr,
    nested_is_not_null_expr, nested_set_expr, overlay_column,
)
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    Error state for PySpark DataFrame validation.
    """

    def __init__(self, error_column: str, with_loc: bool = False) -> None:
        self.error_column = error_column
        self.with_loc = with_loc
        self.names = {}
        self.reserved_names = set()
        self.absent = set()
        self.filled = set()
        self.extras = {}
        # used when casting a map / JSON object to a struct to tell whether a field was in the original object
        self.dynamic_keys = {}
        self.exprs, self.details, self.dropped = {}, {}, set()

    def add_errors(self, boolmask: Column, column: str, details=None, **kwargs) -> None:
        """
        Record an error detail expression for a column (True values in the boolmask indicate errors).
        """

        details = dict(details or kwargs)
        if '[]' not in column:
            self.details.setdefault(column, []).append(fn.when(boolmask, self._details_struct(details)))
            return

        loc = fn.filter(fn.transform(boolmask, fn.when), fn.isnotnull)
        self.details.setdefault(column, []).append(fn.when(fn.array_size(loc) > 0, self._details_struct(details, loc)))

        name = self.indexes_column_for(column)
        indexes = self.exprs.get(name, fn.col(name) if name in self.columns else None)
        self.exprs[name] = boolmask if indexes is None else fn.zip_with(indexes, boolmask, operator.or_)

    def resync(self, df: DataFrame) -> None:
        """
        Refresh the schema knowledge after a rule transformed the DataFrame directly.
        """

        self.columns = set(df.columns)
        self.dtypes = {field.name: field.dataType for field in df.schema.fields}

    def flush(self, df: DataFrame) -> DataFrame:
        """
        Materialize all pending expressions into the DataFrame using a single projection.
        """

        if not (self.exprs or self.details or self.dropped):
            return df

        existing = set(df.columns)
        selection = {name: fn.col(name) for name in df.columns}
        selection.update({name: overlay_column(expr, self.dtypes.get(name)) for name, expr in self.exprs.items()})
        for column, details in self.details.items():
            name = self.details_column_for(column)
            # nulls stay in the error details: filtering them out here would disable the whole-stage codegen
            values = fn.array(*details).cast(self._details_dtype())
            if name not in existing:
                selection[name] = values
            elif details:
                selection[name] = fn.concat(fn.col(name), values)
        for name in self.dropped:
            selection.pop(name, None)

        df = df.select(*(expr.alias(name) for name, expr in selection.items()))
        self.exprs, self.details, self.dropped = {}, {}, set()
        self.resync(df)
        return df

    def expr_of(self, column: str) -> Column:
        """
        Return the current value expression of a column (flattened values for array columns).
        """

        root = self._root_of(column)
        return nested_get_expr(column, self.exprs.get(root), self.dtypes.get(root))

    def dtype_of(self, column: str) -> DataType:
        """
        Return the current data type of a column.
        """

        return dtype_at(self._schema_of(column), column)

    def set_expr(self, column: str, val, dtype: Optional[DataType] = None) -> None:
        """
        Set the value expression of a (possibly nested) column, along with the new data type of the column when it
        changes.
        """

        root = self._root_of(column)
        self.exprs[root] = nested_set_expr(column, val, self.exprs.get(root), self.dtypes.get(root))
        if dtype is not None:
            self.dtypes[root] = dtype_set(self._schema_of(column), column, dtype)[root].dataType

    def drop_column(self, column: str) -> None:
        """
        Drop a (possibly nested) column.
        """

        root = self._root_of(column)
        if '.' not in column:
            self.dropped.add(root)
            self.exprs.pop(root, None)
            self.dtypes.pop(root, None)
            return

        self.exprs[root] = nested_drop_expr(column, self.exprs.get(root, fn.col(root)), self.dtypes.get(root))
        self.dtypes[root] = dtype_drop(self._schema_of(column), column)[root].dataType

    def set_backup(self, column: str) -> None:
        """
        Snapshot the current value expression of a column as its backup.
        """

        self.exprs[self.backup_column_for(column)] = self.expr_of(column)

    def backup_of(self, column: str) -> Column:
        """
        Return the backup expression of a column (or a reference to the materialized backup column).
        """

        name = self.backup_column_for(column)
        return self.exprs.get(name, fn.col(name))

    def presence_of(self, column: str) -> Optional[Column]:
        """
        Return a column presence expression, or None if statically defined by the schema.
        """

        if column in self.filled:
            return None

        parent, _, name = column.rpartition('.')
        keys_of = self.dynamic_keys.get(parent)
        if keys_of is None:
            return None

        return self._elementwise(
            parent,
            lambda container: fn.coalesce(fn.array_contains(keys_of(container), fn.lit(name)), fn.lit(True)),
            self.backup_of(parent),
        )

    def container_exists(self, column: str) -> Column:
        """
        Return a boolean expression (or array of booleans) indicating whether the parent container of the given column
        exists (is not null).
        """

        parent, _, _ = column.rpartition('.')
        if not parent:
            return fn.lit(True)

        root = self._root_of(parent)
        return nested_is_not_null_expr(parent, self.exprs.get(root), self.dtypes.get(root))

    def stub_if_absent(self, column: str) -> None:
        """
        Creating a void column if it is missing.
        """

        try:
            self.dtype_of(column)
        except KeyError:
            self.absent.add(column)
            self.set_expr(column, fn.lit(None).cast('void'), NullType())

    def details_expr(self, details, loc: Optional[Column] = None) -> Column:
        """
        Return a details array holding a single error, shaped like the accumulated ones.
        """

        return fn.array(self._details_struct(details, loc)).cast(self._details_dtype())

    def errors_dtype(self) -> DataType:
        """
        Return the data type of the error map reported to the caller.
        """

        return MapType(StringType(), StructType([
            StructField('details', self._details_dtype(), nullable=False),
            StructField('original', StringType()),
        ]))

    def details_column_for(self, column: str) -> str:
        return f'{self.error_column}_{self._name_for(column)}'

    def indexes_column_for(self, column: str) -> str:
        return f'{self.error_column}_{self._name_for(column)}_idx'

    def backup_column_for(self, column: str) -> str:
        return f'{self.error_column}_{self._name_for(column)}_bak'

    def parsed_column_for(self, column: str) -> str:
        return f'{self.error_column}_{self._name_for(column)}_json'

    def _name_for(self, column: str) -> str:
        """
        Return a deterministic per-column base for service column names, avoiding
        collisions with input columns and with bases already handed out.
        """

        if column in self.names:
            return self.names[column]

        def try_reserve(candidate):
            derived = {f'{self.error_column}_{candidate}{kind}' for kind in ('', '_idx', '_bak', '_json')}
            if derived & self.reserved_names or derived & self.columns:
                return False
            self.reserved_names |= derived
            return True

        base = column.replace('[]', '_arr').replace('.', '_')
        name, attempt = base, 1
        while not try_reserve(name):
            attempt += 1
            name = f'{base}_{attempt}'
        self.names[column] = name
        return name

    def _schema_of(self, column: str) -> StructType:
        root = self._root_of(column)
        return StructType([StructField(root, self.dtypes[root])] if root in self.dtypes else [])

    def _details_struct(self, details, loc: Optional[Column] = None) -> Column:
        fields = [fn.lit(value).alias(name) for name, value in dict(details).items()]
        if self.with_loc:
            # if DataFrame has at least one array column, all details carry indexes
            # because we cannot append a struct to an array with different types
            fields.append((fn.lit(None).cast('array<int>') if loc is None else loc).alias('loc'))
        return fn.struct(*fields)

    def _details_dtype(self) -> DataType:
        fields = [StructField('type', StringType()), StructField('msg', StringType())]
        if self.with_loc:
            fields.append(StructField('loc', ArrayType(IntegerType())))
        return ArrayType(StructType(fields))

    @staticmethod
    def _root_of(column: str) -> str:
        return column.partition('.')[0].removesuffix('[]')

    @staticmethod
    def _elementwise(column: str, func: Callable[..., Column], *cols: Column) -> Column:
        if '[]' not in column:
            return func(*cols)
        return fn.transform(*cols, func) if len(cols) == 1 else fn.zip_with(*cols, func)


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


class RequiredRule(Rule):
    """
    Rule reporting a column that the schema declares and the DataFrame schema does not carry
    """

    accumulates = True

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        presence = error_state.presence_of(column)
        if presence is not None:
            boolmask = fn.transform(presence, operator.inv) if '[]' in column else ~presence
        elif column in error_state.absent:
            boolmask = error_state.container_exists(column)
        else:
            return df

        error_state.add_errors(boolmask, column, errors.MISSING)
        return df


def required() -> Rule:
    return RequiredRule()


class DefaultRule(Rule):
    """
    Rule injecting a static value into the slots the input never populates
    """

    accumulates = True

    def __init__(self, value: Any) -> None:
        self.value = value

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        presence = error_state.presence_of(column)
        if presence is None:
            if column not in error_state.absent:
                return df

            literal = fn.lit(self.value)
            error_state.set_expr(column, literal, dtype=df.select(literal.alias('_')).schema[0].dataType)
            error_state.absent.discard(column)
            return df

        literal = fn.lit(self.value).cast(error_state.dtype_of(column))
        error_state.filled.add(column)
        if '[]' not in column:
            error_state.set_expr(column, fn.when(~presence, literal).otherwise(error_state.expr_of(column)))
            return df

        def fill(value: Column, present: Column, nodes: Iterable[str], literal: Column) -> Column:
            if not nodes:
                return fn.when(~present, literal).otherwise(value)

            head, *rest = nodes
            return value.withField(head, fill(value.getField(head), present, rest, literal))

        root, _, nested = column.partition('[].')
        root = root.removesuffix('[]')
        nodes = nested.split('.') if nested else []
        error_state.set_expr(
            root,
            fn.zip_with(error_state.expr_of(root), presence, lambda x, y: fill(x, y, nodes, literal)),
        )
        return df


def default(value: Any) -> Rule:
    return DefaultRule(value)


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
        boolmask = error_state.container_exists(column)
        error_state.drop_column(column.removesuffix('[]'))
        error_state.add_errors(boolmask, column, errors.EXTRA_FORBIDDEN)
        return df


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    return ExtraForbiddenRule(allowed)


def int_parsing(nullable: bool = False) -> Rule:
    return DataTypeRule(
        lambda col: col.cast('integer'),
        IntegerType,
        (StringType, NumericType, BooleanType),
        errors.INT_TYPE,
        errors.INT_PARSING,
        nullable=nullable,
    )


def float_parsing(nullable: bool = False):
    return DataTypeRule(
        lambda col: col.cast('float'),
        FloatType,
        (StringType, NumericType, BooleanType),
        errors.FLOAT_TYPE,
        errors.FLOAT_PARSING,
        nullable=nullable,
    )


def str_parsing(nullable: bool = False) -> Rule:
    return DataTypeRule(
        lambda col: col.cast('string'),
        StringType,
        (NumericType, BooleanType, DateType, TimestampType),
        errors.STRING_TYPE,
        nullable=nullable,
    )


def bool_parsing(nullable: bool = False) -> Rule:
    return BooleanTypeRule(
        lambda col: col.cast('boolean'),
        BooleanType,
        (NumericType, StringType),
        errors.BOOL_TYPE,
        errors.BOOL_PARSING,
        nullable=nullable,
    )


def datetime_parsing(nullable: bool = False) -> Rule:
    return DataTypeRule(
        lambda col: fn.to_timestamp(col),  # noqa: PLW0108
        TimestampType,
        (StringType, DateType),
        errors.DATETIME_TYPE,
        errors.DATETIME_PARSING,
        nullable=nullable,
    )


def date_parsing(nullable: bool = False) -> Rule:
    return DataTypeRule(
        lambda col: fn.to_date(fn.to_timestamp(col)),
        DateType,
        (StringType, TimestampType),
        errors.DATE_TYPE,
        errors.DATE_PARSING,
        nullable=nullable,
    )


def time_parsing(nullable: bool = False) -> Rule:
    time_regex = (
        r'^(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{1,6})?(Z|[+-](2[0-3]|[01][0-9]):([0-5][0-9]))?$'
    )
    return ComplexTypeRule(
        lambda col: fn.when(col.rlike(time_regex), col),
        StringType,
        (StringType,),
        errors.TIME_TYPE,
        errors.TIME_PARSING,
        nullable=nullable,
    )


def duration_parsing(nullable: bool = False) -> Rule:
    duration_regex = r'^-?P(?=\d|T\d)(\d+Y)?(\d+M)?(\d+D)?(T(?=\d)(\d+H)?(\d+M)?(\d+(\.\d+)?S)?)?$'
    return ComplexTypeRule(
        lambda col: fn.when(col.rlike(duration_regex), col),
        StringType,
        (StringType,),
        errors.DURATION_TYPE,
        errors.DURATION_PARSING,
        nullable=nullable,
    )


def uuid_parsing(nullable: bool = False) -> Rule:
    uuid_regex = '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    return ComplexTypeRule(
        lambda col: fn.when(fn.lower(col).rlike(uuid_regex), fn.lower(col)),
        StringType,
        (StringType,),
        errors.UUID_TYPE,
        errors.UUID_PARSING,
        nullable=nullable,
    )


def ipv4_address(nullable: bool = False) -> Rule:
    ipv4_regex = r'^((25[0-5]|2[0-4]\d|(1\d{2}|[1-9]\d|\d))\.){3}(25[0-5]|2[0-4]\d|(1\d{2}|[1-9]\d|\d))$'
    return ComplexTypeRule(
        lambda col: fn.when(col.rlike(ipv4_regex), col),
        StringType,
        (StringType,),
        errors.IPV4,
        errors.IPV4,
        nullable=nullable,
    )


def ipv6_address(nullable: bool = False) -> Rule:
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
        nullable=nullable,
    )


def uri_parsing(nullable: bool = False) -> Rule:
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
        nullable=nullable,
    )


def object_parsing(schema, nullable: bool = False) -> Rule:
    return ObjectTypeRule(schema, nullable)


def array_parsing(schema, nullable: bool = False) -> Rule:
    return ArrayTypeRule(schema, nullable)


class DataTypeRule(Rule):
    """
    Abstract base class for casting to target data type
    """

    accumulates = True

    def __init__(  # noqa: PLR0913
        self, caster, dtype, supported_cast_dtypes,
        type_error_details=None, parsing_error_details=None, *, nullable=True,
    ):
        self.caster = caster
        self.dtype = dtype
        self.supported_cast_dtypes = supported_cast_dtypes
        self.type_error_details = type_error_details
        self.parsing_error_details = parsing_error_details
        self.nullable = nullable

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        if not self.nullable and column not in error_state.absent:
            self._flag_nulls(column, error_state)

        data_type = error_state.dtype_of(column)
        if self._is_valid_dtype(data_type):
            return df

        if data_type.typeName() == 'void':
            self._mark_fields_absent(column, error_state)
            error_state.set_expr(column, fn.lit(None).cast(self._target_dtype()), self._target_dtype())
            return df

        if '[]' in column:
            return self._verify_array(df, column, data_type, error_state)
        return self._verify_scalar(df, column, data_type, error_state)

    def _verify_scalar(self, df: DataFrame, column: str, data_type: DataType, error_state: ErrorState) -> DataFrame:
        error_state.set_backup(column)
        if not isinstance(data_type, self.supported_cast_dtypes):
            error_state.set_expr(column, fn.lit(None).cast(self._target_dtype()), self._target_dtype())
            error_state.add_errors(fn.lit(True), column, self.type_error_details)
            return df

        self._cast_scalar(column, data_type, error_state)
        if not self.parsing_error_details:
            return df

        error_state.add_errors(
            fn.isnull(error_state.expr_of(column)) & fn.isnotnull(error_state.backup_of(column)),
            column,
            self.parsing_error_details,
        )
        return df

    def _verify_array(self, df: DataFrame, column: str, element_type: DataType, error_state: ErrorState) -> DataFrame:
        error_state.set_backup(column)
        if not isinstance(element_type, self.supported_cast_dtypes):
            indexes = fn.transform(error_state.expr_of(column.split('[]', maxsplit=1)[0]), lambda _: fn.lit(True))
            error_state.set_expr(column, fn.lit(None).cast(self._target_dtype()), self._target_dtype())
            error_state.add_errors(indexes, column, self.type_error_details)
            return df

        self._cast_array(column, element_type, error_state)
        if not self.parsing_error_details:
            return df

        actual_values = error_state.expr_of(column)
        indexes = fn.zip_with(
            actual_values, error_state.backup_of(column), lambda x, y: fn.isnull(x) & fn.isnotnull(y)
        )
        error_state.add_errors(indexes, column, self.parsing_error_details)
        return df

    def _mark_fields_absent(self, column: str, error_state: ErrorState) -> None:
        pass

    def _flag_nulls(self, column: str, error_state: ErrorState) -> None:
        error_state.set_backup(column)
        expr = error_state.expr_of(column)
        parent = '' if column.endswith('[]') else column.rpartition('.')[0]

        if '[]' not in column:
            boolmask = fn.isnull(expr)
            if parent:
                boolmask &= error_state.container_exists(column)
        elif parent:
            boolmask = fn.zip_with(expr, error_state.container_exists(column), lambda x, y: fn.isnull(x) & y)
        else:
            boolmask = fn.transform(expr, fn.isnull)

        presence = error_state.presence_of(column)
        if presence is not None:
            boolmask = fn.zip_with(boolmask, presence, operator.and_) if '[]' in column else boolmask & presence

        error_state.add_errors(boolmask, column, self.type_error_details)

    def _is_valid_dtype(self, data_type) -> bool:
        return isinstance(data_type, self.dtype)

    def _target_dtype(self):
        return self.dtype()

    def _cast_result_dtype(self, data_type):  # noqa: ARG002
        return self._target_dtype()

    def _cast_scalar(self, column, data_type, error_state):
        error_state.set_expr(column, self.caster(error_state.expr_of(column)), self._cast_result_dtype(data_type))

    def _cast_array(self, column, element_type, error_state):
        error_state.set_expr(column, self.caster, self._cast_result_dtype(element_type))


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

    def __init__(self, schema, nullable=True):
        super().__init__(
            None, StructType, (StringType, MapType), errors.OBJECT_TYPE, errors.OBJECT_PARSING, nullable=nullable
        )
        self.struct_type = self.parse_struct_type(schema)
        self.allow_extra = schema.allow_extra

    def _is_valid_dtype(self, data_type) -> bool:
        return isinstance(data_type, StructType)

    def _verify_scalar(self, df: DataFrame, column: str, data_type: DataType, error_state: ErrorState) -> DataFrame:
        if isinstance(data_type, StringType):
            # the json text is parsed once into a service column: an inlined from_json would be evaluated again by
            # every expression reading one of the fields
            parsed = error_state.parsed_column_for(column)
            error_state.exprs[parsed] = fn.from_json(error_state.expr_of(column), MapType(StringType(), StringType()))
            df = error_state.flush(df)

        return super()._verify_scalar(df, column, data_type, error_state)

    def _mark_fields_absent(self, column: str, error_state: ErrorState) -> None:
        for name in self.struct_type.fieldNames():
            error_state.absent.add(f'{column}.{name}')

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
        dtype = self._cast_result_dtype(data_type)
        if isinstance(data_type, MapType):
            error_state.dynamic_keys[column] = fn.map_keys
            self._flag_extra_keys(column, data_type, error_state)
            error_state.set_expr(column, self._struct_of(error_state.expr_of(column)), dtype)
            return

        parsed = fn.col(error_state.parsed_column_for(column))
        error_state.dynamic_keys[column] = lambda _: fn.map_keys(parsed)
        self._flag_extra_keys(column, data_type, error_state)
        error_state.set_expr(column, self._struct_of(parsed), dtype)

    def _cast_array(self, column, element_type, error_state):
        dtype = self._cast_result_dtype(element_type)
        if isinstance(element_type, MapType):
            error_state.dynamic_keys[column] = fn.map_keys
            self._flag_extra_element_keys(column, element_type, error_state)
            error_state.set_expr(column, lambda item: self._struct_of(item).column(dtype), dtype)
            return

        error_state.dynamic_keys[column] = fn.json_object_keys
        self._flag_extra_element_keys(column, element_type, error_state)
        error_state.set_expr(column, lambda item: self._parse_json(item).column(dtype), dtype)

    def _parse_json(self, col_expr: Column) -> StructOverlay:
        return self._struct_of(fn.from_json(col_expr, MapType(StringType(), StringType())))

    def _struct_of(self, col_expr: Column) -> StructOverlay:
        # every field reads its own key: a single struct expression would be inlined into each field reference,
        # growing the plan quadratically with the number of fields
        overlay = StructOverlay(col_expr)
        overlay.fields.update({field: col_expr[field] for field in self.struct_type.fieldNames()})
        return overlay

    def _flag_extra_keys(self, column, data_type, error_state) -> None:
        if self.allow_extra:
            return

        extras = self._extras_of(fn.col(error_state.backup_column_for(column)), data_type)
        keys = fn.coalesce(fn.map_keys(extras), fn.array().cast('array<string>'))

        error_state.extras[column] = fn.map_from_arrays(
            fn.transform(keys, lambda key: fn.concat(fn.lit(f'{column}.'), key)),
            fn.transform(keys, lambda key: fn.struct(
                error_state.details_expr(errors.EXTRA_FORBIDDEN).alias('details'),
                fn.element_at(extras, key).alias('original'),
            )),
        ).cast(error_state.errors_dtype())

    def _flag_extra_element_keys(self, column, element_type, error_state) -> None:
        if self.allow_extra:
            return

        extras = self._extras_of(fn.col(error_state.backup_column_for(column)), element_type, elementwise=True)
        keys = fn.array_distinct(fn.coalesce(
            fn.flatten(
                fn.transform(extras, lambda item: fn.coalesce(fn.map_keys(item), fn.array().cast('array<string>')))),
            fn.array().cast('array<string>')
        ))

        def loc_of(key):
            return fn.filter(
                fn.transform(extras, lambda item, i: fn.when(fn.array_contains(fn.map_keys(item), key), i)),
                fn.isnotnull,
            )

        def origin_of(key):
            values = fn.transform(extras, lambda item: fn.coalesce(fn.element_at(item, key), fn.lit('null')))
            return fn.concat(fn.lit('['), fn.array_join(values, ','), fn.lit(']'))

        error_state.extras[column] = fn.map_from_arrays(
            fn.transform(keys, lambda key: fn.concat(fn.lit(f'{column}.'), key)),
            fn.transform(keys, lambda key: fn.struct(
                error_state.details_expr(errors.EXTRA_FORBIDDEN, loc_of(key)).alias('details'),
                origin_of(key).alias('original'),
            )),
        ).cast(error_state.errors_dtype())

    def _extras_of(self, col_expr: Column, value_type: DataType, elementwise: bool = False) -> Column:
        allowed = frozenset(self.struct_type.fieldNames())
        if isinstance(value_type, MapType):
            names = fn.array(*[fn.lit(name) for name in allowed]).cast('array<string>')

            def extras_of(item):
                return fn.transform_values(
                    fn.map_filter(item, lambda key, _: ~fn.array_contains(names, key)),
                    lambda _, value: format_value(value, value_type.valueType.typeName())
                )

            return fn.transform(col_expr, extras_of) if elementwise else extras_of(col_expr)

        def _extra_keys(text, allowed_keys):
            try:
                value = json.loads(text)
            except (TypeError, ValueError):
                return None

            if not isinstance(value, dict):
                return None

            return {
                key: json.dumps(item, separators=(',', ':'))
                for key, item in value.items()
                if key not in allowed_keys
            }

        if elementwise:
            # a python udf cannot be called from inside a higher order function, so the whole array is handed over
            @fn.udf(returnType=ArrayType(MapType(StringType(), StringType())))
            def _json_extras(texts):
                return None if texts is None else [_extra_keys(text, allowed) for text in texts]
        else:
            @fn.udf(returnType=MapType(StringType(), StringType()))
            def _json_extras(text):
                return _extra_keys(text, allowed)

        return _json_extras(col_expr)

    @staticmethod
    def parse_struct_type(schema) -> StructType:
        return StructType([StructField(name, StringType(), nullable=True, metadata={}) for name in schema.columns])


class ArrayTypeRule(DataTypeRule):
    """
    Class for casting to array type
    """

    def __init__(self, schema, nullable=True):
        super().__init__(
            None, ArrayType, (StringType, ArrayType), errors.ARRAY_TYPE, errors.ARRAY_PARSING, nullable=nullable
        )
        self.struct_type = self.parse_array_type(schema)

    def _verify_array(self, df: DataFrame, column: str, element_type: DataType, error_state: ErrorState) -> DataFrame:
        raise NotImplementedError('nested arrays are not supported yet')

    def _mark_fields_absent(self, column: str, error_state: ErrorState) -> None:
        element_type = self.struct_type.elementType
        if isinstance(element_type, StructType):
            for name in element_type.fieldNames():
                error_state.absent.add(f'{column}[].{name}')

    def _target_dtype(self):
        return self.struct_type

    def _cast_scalar(self, column, data_type, error_state):
        # from_json reads a bare object as a single element array, which the schema did not ask for: only
        # text that opens as an array is parsed, and anything else is left to the parsing error to report
        expr = error_state.expr_of(column)
        parsed = fn.when(expr.rlike(r'^\s*\['), fn.from_json(expr, self.struct_type))
        error_state.set_expr(column, parsed, self._cast_result_dtype(data_type))

    @staticmethod
    def parse_array_type(schema) -> ArrayType:
        if schema.inner_schema is None:
            return ArrayType(StringType())
        if schema.dtype in {'array', 'list'}:
            raise NotImplementedError('nested arrays are not supported yet')
        return ArrayType(ObjectTypeRule.parse_struct_type(schema.inner_schema))
