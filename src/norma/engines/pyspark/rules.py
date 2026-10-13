import functools  # noqa: I001
import inspect
import json
import operator
from typing import Any, Iterable, Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import (
    ArrayType, BooleanType, DataType, DateType, FloatType, IntegerType, MapType, NullType, NumericType, StringType,
    StructField, StructType, TimestampType,
)

from norma import errors
from norma.engines.pyspark.utils import (
    PendingStruct, dtype_at, dtype_drop, dtype_set, elementwise, format_value, materialize, nested_drop_expr,
    nested_get_expr, nested_is_not_null_expr, nested_set_expr,
)
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule

# the type a JSON object is parsed into, before its fields are cast one by one
_JSON_OBJECT = MapType(StringType(), StringType())


class ErrorState(IErrorState):
    """
    Error state for PySpark DataFrame validation.

    :param df: The DataFrame to validate
    :param error_column: The name of the column to store error information
    :param with_loc: Whether error details carry the 'loc' field with positions of invalid array items
    """

    def __init__(self, df: DataFrame, error_column: str, with_loc: bool = False) -> None:
        self.error_column = error_column
        self.with_loc = with_loc
        self.names = {}
        self.reserved_names = set()
        self.absent = set()
        self.filled = set()
        self.extras = {}
        self.original_objects = {}
        self.exprs, self.details, self.dropped = {}, {}, set()
        self.resync(df)

    def add_errors(self, boolmask: Column, column: str, details) -> None:
        """
        Record an error detail expression for a column (True values in the boolmask indicate errors).
        """

        if '[]' not in column:
            self.details.setdefault(column, []).append(fn.when(boolmask, fn.array(self._details_struct(details))))
            return

        loc = fn.filter(fn.transform(boolmask, fn.when), fn.isnotnull)
        self.details.setdefault(column, []).append(
            fn.when(fn.array_size(loc) > 0, fn.array(self._details_struct(details, loc)))
        )

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
        for name, expr in self.exprs.items():
            selection[name] = materialize(expr, self.dtypes.get(name)).alias(name)
        empty = fn.array()
        for column, details in self.details.items():
            name = self.details_column_for(column)
            values = fn.concat(*(detail.otherwise(empty) for detail in details)).cast(self._details_dtype())
            if name in existing:
                values = fn.concat(fn.col(name), values)
            selection[name] = values.alias(name)
        for name in self.dropped:
            selection.pop(name, None)

        df = df.select(*selection.values())
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

    def has_column(self, column: str) -> bool:
        """
        Return whether a (possibly nested) column exists.
        """

        try:
            self.dtype_of(column)
        except KeyError:
            return False
        return True

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
        objects = self.original_objects.get(parent)
        if objects is None:
            return None

        return elementwise(
            parent,
            lambda container: fn.coalesce(fn.array_contains(fn.map_keys(container), fn.lit(name)), fn.lit(True)),
            self.exprs.get(objects, fn.col(objects)),
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

        if not self.has_column(column):
            self.absent.add(column)
            self.set_expr(column, fn.lit(None).cast('void'), NullType())

    def add_extra_keys(self, column: str, keys: Column, original_of, loc_of=None) -> None:
        """
        Record an error for every key of a map / JSON object cast to a struct that does not declare it.
        """

        def details_of(key):
            loc = None if loc_of is None else loc_of(key)
            return fn.array(self._details_struct(errors.EXTRA_FORBIDDEN, loc)).cast(self._details_dtype())

        self.extras[column] = fn.map_from_arrays(
            fn.transform(keys, lambda key: fn.concat(fn.lit(f'{column}.'), key)),
            fn.transform(keys, lambda key: fn.struct(
                details_of(key).alias('details'),
                original_of(key).alias('original'),
            )),
        ).cast(self.errors_dtype())

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


class BaseRule(Rule):
    """
    Base rule class for PySpark DataFrame validation

    :param func: The function returning the error mask
    :param details: The details of the reported errors
    """

    accumulates = True

    def __init__(self, func, details=None):
        self.func = func
        self.details = details

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        """
        Verify the DataFrame against the rule
        """

        params = inspect.signature(self.func).parameters
        if '[]' in column and 'df' not in params:
            error_state.add_errors(fn.transform(error_state.expr_of(column), self.func), column, self.details)
            return df

        if params.keys() & {'df', 'col', 'column'}:
            df = error_state.flush(df)

        col_expr = error_state.expr_of(column)
        values = {
            'df': df, 'col': column, 'column': column, 'col_expr': col_expr, 'column_expr': col_expr,
            'error_state': error_state,
        }
        result = self.func(**{param: value for param, value in values.items() if param in params})
        if 'df' in params:
            # the rule transformed the DataFrame directly
            error_state.resync(result)
            return result

        error_state.add_errors(result, column, self.details)
        return df


def rule(func, details=None) -> BaseRule:
    return BaseRule(func, details)


class RequiredRule(Rule):
    """
    Rule reporting a column that the schema declares and the DataFrame schema does not carry
    """

    accumulates = True

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        presence = error_state.presence_of(column)
        if presence is not None:
            boolmask = elementwise(column, operator.inv, presence)
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


def applies_to(dtype: type[DataType]):
    """
    Decorate a rule factory, so that its rules can only be applied to columns of the given data type
    """

    def decorator(factory):
        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            instance = factory(*args, **kwargs)
            verify = instance.verify

            def checked_verify(df, column, error_state):
                if not isinstance(error_state.dtype_of(column), dtype):
                    raise ValueError(f'{factory.__name__} rule can only be applied to {dtype.typeName()} columns')
                return verify(df, column, error_state)

            instance.verify = checked_verify
            return instance

        return wrapper

    return decorator


@applies_to(NumericType)
def multiple_of(multiple: float) -> Rule:
    if multiple <= 0:
        raise ValueError('multiple_of must be greater than zero')

    return rule(
        lambda col_expr: (col_expr < fn.lit(0)) | ((col_expr % fn.lit(multiple)) != fn.lit(0)),
        details=errors.MULTIPLE_OF.format(multiple_of=multiple),
    )


@applies_to(StringType)
def min_length(value: int) -> Rule:
    return rule(
        lambda col_expr: fn.length(col_expr) < value,
        details=errors.STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else ''),
    )


@applies_to(StringType)
def max_length(value: int) -> Rule:
    return rule(
        lambda col_expr: fn.length(col_expr) > value,
        details=errors.STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else ''),
    )


@applies_to(StringType)
def pattern(regex: str) -> Rule:
    return rule(
        lambda col_expr: ~col_expr.rlike(regex),
        details=errors.STRING_PATTERN_MISMATCH.format(pattern=regex),
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


@applies_to(ArrayType)
def unique_items() -> Rule:
    return rule(
        lambda col_expr: fn.size(col_expr) != fn.size(fn.array_distinct(col_expr)),
        details=errors.UNIQUE_ITEMS,
    )


@applies_to(ArrayType)
def max_items(value: int) -> Rule:
    return rule(
        lambda col_expr: fn.array_size(col_expr) > value,
        details=errors.TOO_LONG.format(_type_='Array', max_length=value, _plural_='s' if value > 1 else ''),
    )


@applies_to(ArrayType)
def min_items(value: int) -> Rule:
    return rule(
        lambda col_expr: fn.array_size(col_expr) < value,
        details=errors.TOO_SHORT.format(_type_='Array', min_length=value, _plural_='s' if value > 1 else ''),
    )


class ExtraForbiddenRule(Rule):
    """
    Rule reporting a column that the schema does not declare, moving its values to a backup column
    """

    accumulates = True

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        error_state.set_backup(column)
        boolmask = error_state.container_exists(column)
        error_state.drop_column(column)
        error_state.add_errors(boolmask, column, errors.EXTRA_FORBIDDEN)
        return df


def extra_forbidden() -> Rule:
    return ExtraForbiddenRule()


def int_parsing(nullable: bool = False) -> Rule:
    return DataTypeRule(
        lambda col: col.cast('integer'),
        IntegerType(),
        (StringType, NumericType, BooleanType),
        errors.INT_TYPE,
        errors.INT_PARSING,
        nullable=nullable,
    )


def float_parsing(nullable: bool = False):
    return DataTypeRule(
        lambda col: col.cast('float'),
        FloatType(),
        (StringType, NumericType, BooleanType),
        errors.FLOAT_TYPE,
        errors.FLOAT_PARSING,
        nullable=nullable,
    )


def str_parsing(nullable: bool = False) -> Rule:
    return DataTypeRule(
        lambda col: col.cast('string'),
        StringType(),
        (NumericType, BooleanType, DateType, TimestampType),
        errors.STRING_TYPE,
        nullable=nullable,
    )


def bool_parsing(nullable: bool = False) -> Rule:
    return BooleanTypeRule(
        lambda col: col.cast('boolean'),
        BooleanType(),
        (NumericType, StringType),
        errors.BOOL_TYPE,
        errors.BOOL_PARSING,
        nullable=nullable,
    )


def datetime_parsing(nullable: bool = False) -> Rule:
    return DataTypeRule(
        lambda col: fn.to_timestamp(col),  # noqa: PLW0108
        TimestampType(),
        (StringType, DateType),
        errors.DATETIME_TYPE,
        errors.DATETIME_PARSING,
        nullable=nullable,
    )


def date_parsing(nullable: bool = False) -> Rule:
    return DataTypeRule(
        lambda col: fn.to_date(fn.to_timestamp(col)),
        DateType(),
        (StringType, TimestampType),
        errors.DATE_TYPE,
        errors.DATE_PARSING,
        nullable=nullable,
    )


def time_parsing(regex: str, nullable: bool = False) -> Rule:
    return StringDerivedTypeRule(
        lambda col: fn.when(col.rlike(regex), col),
        StringType(),
        (StringType,),
        errors.TIME_TYPE,
        errors.TIME_PARSING,
        nullable=nullable,
    )


def duration_parsing(regex: str, nullable: bool = False) -> Rule:
    return StringDerivedTypeRule(
        lambda col: fn.when(col.rlike(regex), col),
        StringType(),
        (StringType,),
        errors.DURATION_TYPE,
        errors.DURATION_PARSING,
        nullable=nullable,
    )


def uuid_parsing(regex: str, nullable: bool = False) -> Rule:
    return StringDerivedTypeRule(
        lambda col: fn.when(fn.lower(col).rlike(regex), fn.lower(col)),
        StringType(),
        (StringType,),
        errors.UUID_TYPE,
        errors.UUID_PARSING,
        nullable=nullable,
    )


def ipv4_address(regex: str, nullable: bool = False) -> Rule:
    return StringDerivedTypeRule(
        lambda col: fn.when(col.rlike(regex), col),
        StringType(),
        (StringType,),
        errors.IPV4,
        errors.IPV4,
        nullable=nullable,
    )


def ipv6_address(regex: str, nullable: bool = False) -> Rule:
    return StringDerivedTypeRule(
        lambda col: fn.when(col.rlike(regex), col),
        StringType(),
        (StringType,),
        errors.IPV6,
        errors.IPV6,
        nullable=nullable,
    )


def uri_parsing(regex: str, nullable: bool = False) -> Rule:
    return StringDerivedTypeRule(
        lambda col: fn.when(col.rlike(regex), col),
        StringType(),
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
    Rule casting a column to the target data type, reporting the values that cannot be cast.
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

        if isinstance(data_type, NullType):
            self._mark_fields_absent(column, error_state)
            error_state.set_expr(column, fn.lit(None).cast(self.dtype), self.dtype)
            return df

        return self._verify_cast(df, column, data_type, error_state)

    def _verify_cast(self, df: DataFrame, column: str, data_type: DataType, error_state: ErrorState) -> DataFrame:
        error_state.set_backup(column)
        if not isinstance(data_type, self.supported_cast_dtypes):
            mismatch = elementwise(column, lambda _: fn.lit(True), error_state.expr_of(column.partition('[]')[0]))
            error_state.set_expr(column, fn.lit(None).cast(self.dtype), self.dtype)
            error_state.add_errors(mismatch, column, self.type_error_details)
            return df

        self._cast(column, data_type, error_state)
        if not self.parsing_error_details:
            return df

        failed = elementwise(
            column,
            lambda value, original: fn.isnull(value) & fn.isnotnull(original),
            error_state.expr_of(column),
            error_state.backup_of(column),
        )
        error_state.add_errors(failed, column, self.parsing_error_details)
        return df

    def _mark_fields_absent(self, column: str, error_state: ErrorState) -> None:
        pass

    def _flag_nulls(self, column: str, error_state: ErrorState) -> None:
        error_state.set_backup(column)
        expr = error_state.expr_of(column)

        if column.endswith('[]') or '.' not in column:
            boolmask = elementwise(column, fn.isnull, expr)
        else:
            # a null field of a missing parent is not reported
            boolmask = elementwise(column, lambda x, y: fn.isnull(x) & y, expr, error_state.container_exists(column))

        presence = error_state.presence_of(column)
        if presence is not None:
            boolmask = elementwise(column, operator.and_, boolmask, presence)

        error_state.add_errors(boolmask, column, self.type_error_details)

    def _is_valid_dtype(self, data_type: DataType) -> bool:
        return isinstance(data_type, type(self.dtype))

    def _cast_result_dtype(self, data_type: DataType) -> DataType:  # noqa: ARG002
        return self.dtype

    def _cast(self, column: str, data_type: DataType, error_state: ErrorState) -> None:
        error_state.set_expr(column, self.caster, self._cast_result_dtype(data_type))


class StringDerivedTypeRule(DataTypeRule):
    """
    Rule keeping the strings that match a format: a column is checked even when it already holds strings
    """

    def _is_valid_dtype(self, data_type: DataType) -> bool:  # noqa: ARG002
        return False


class BooleanTypeRule(DataTypeRule):
    """
    Rule casting to boolean type, reading 'on' / 'off' strings as well
    """

    def _cast(self, column: str, data_type: DataType, error_state: ErrorState) -> None:
        def cast_str(col):
            value = fn.lower(fn.trim(col))
            return fn.when(value.isin(['on', 'off']), value == 'on').otherwise(col.cast('boolean')).cast('boolean')

        error_state.set_expr(
            column, cast_str if isinstance(data_type, StringType) else self.caster, self._cast_result_dtype(data_type)
        )


class ObjectTypeRule(DataTypeRule):
    """
    Rule casting a map or a JSON string to the struct declared by the schema
    """

    def __init__(self, schema, nullable=True):
        super().__init__(
            None, self.parse_struct_type(schema), (StringType, MapType), errors.OBJECT_TYPE, errors.OBJECT_PARSING,
            nullable=nullable,
        )
        self.allow_extra = schema.allow_extra

    def _verify_cast(self, df: DataFrame, column: str, data_type: DataType, error_state: ErrorState) -> DataFrame:
        if isinstance(data_type, StringType):
            parsed = error_state.parsed_column_for(column)
            error_state.exprs[parsed] = elementwise(
                column, lambda text: fn.from_json(text, _JSON_OBJECT), error_state.expr_of(column)
            )
            df = error_state.flush(df)

        return super()._verify_cast(df, column, data_type, error_state)

    def _mark_fields_absent(self, column: str, error_state: ErrorState) -> None:
        for name in self.dtype.fieldNames():
            error_state.absent.add(f'{column}.{name}')

    def _cast_result_dtype(self, data_type: DataType) -> DataType:
        if isinstance(data_type, MapType):
            return StructType([
                StructField(name, data_type.valueType, nullable=True, metadata={}) for name in self.dtype.fieldNames()
            ])
        return self.dtype

    def _cast(self, column: str, data_type: DataType, error_state: ErrorState) -> None:
        if isinstance(data_type, MapType):
            objects = error_state.backup_column_for(column)
        else:
            objects = error_state.parsed_column_for(column)
        error_state.original_objects[column] = objects
        self._flag_extra_keys(column, data_type, error_state)

        dtype = self._cast_result_dtype(data_type)
        if '[]' not in column:
            value = error_state.expr_of(column) if isinstance(data_type, MapType) else fn.col(objects)
            error_state.set_expr(column, self._struct_of(value), dtype)
        elif isinstance(data_type, MapType):
            error_state.set_expr(column, lambda item: self._struct_of(item).materialize(dtype), dtype)
        else:
            parsed = fn.col(objects)
            error_state.set_expr(column, lambda _, i: self._struct_of(parsed[i]).materialize(dtype), dtype)

    def _struct_of(self, col_expr: Column) -> PendingStruct:
        pending = PendingStruct(col_expr)
        pending.fields.update({field: col_expr[field] for field in self.dtype.fieldNames()})
        return pending

    def _flag_extra_keys(self, column: str, data_type: DataType, error_state: ErrorState) -> None:
        if self.allow_extra:
            return

        extras = self._extras_of(column, fn.col(error_state.backup_column_for(column)), data_type)
        if '[]' not in column:
            keys = fn.coalesce(fn.map_keys(extras), fn.array().cast('array<string>'))
            error_state.add_extra_keys(column, keys, lambda key: fn.element_at(extras, key))
            return

        keys = fn.array_distinct(fn.coalesce(
            fn.flatten(
                fn.transform(extras, lambda item: fn.coalesce(fn.map_keys(item), fn.array().cast('array<string>')))),
            fn.array().cast('array<string>')
        ))

        def original_of(key):
            values = fn.transform(extras, lambda item: fn.coalesce(fn.element_at(item, key), fn.lit('null')))
            return fn.concat(fn.lit('['), fn.array_join(values, ','), fn.lit(']'))

        def loc_of(key):
            return fn.filter(
                fn.transform(extras, lambda item, i: fn.when(fn.array_contains(fn.map_keys(item), key), i)),
                fn.isnotnull,
            )

        error_state.add_extra_keys(column, keys, original_of, loc_of)

    def _extras_of(self, column: str, col_expr: Column, value_type: DataType) -> Column:
        """
        Return the keys that the struct does not declare with their original values (one map per element for array
        columns).
        """

        allowed = frozenset(self.dtype.fieldNames())
        if isinstance(value_type, MapType):
            names = fn.array(*[fn.lit(name) for name in self.dtype.fieldNames()]).cast('array<string>')

            def extras_of(item):
                return fn.transform_values(
                    fn.map_filter(item, lambda key, _: ~fn.array_contains(names, key)),
                    lambda _, value: format_value(value, value_type.valueType.typeName())
                )

            return elementwise(column, extras_of, col_expr)

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

        if '[]' in column:
            # a python udf cannot be called from inside a higher order function, so the whole array is handed over
            @fn.udf(returnType=ArrayType(_JSON_OBJECT))
            def _json_extras(texts):
                return None if texts is None else [_extra_keys(text, allowed) for text in texts]
        else:
            @fn.udf(returnType=_JSON_OBJECT)
            def _json_extras(text):
                return _extra_keys(text, allowed)

        return _json_extras(col_expr)

    @staticmethod
    def parse_struct_type(schema) -> StructType:
        return StructType([StructField(name, StringType(), nullable=True, metadata={}) for name in schema.columns])


class ArrayTypeRule(DataTypeRule):
    """
    Rule casting a JSON string to the array declared by the schema
    """

    def __init__(self, schema, nullable=True):
        dtype = self.parse_array_type(schema)
        super().__init__(
            # from_json reads a bare object as a single element array, which the schema did not ask for: only
            # text that opens as an array is parsed, and anything else is left to the parsing error to report
            lambda col: fn.when(col.rlike(r'^\s*\['), fn.from_json(col, dtype)),
            dtype, (StringType, ArrayType), errors.ARRAY_TYPE, errors.ARRAY_PARSING, nullable=nullable,
        )

    def _mark_fields_absent(self, column: str, error_state: ErrorState) -> None:
        element_type = self.dtype.elementType
        if isinstance(element_type, StructType):
            for name in element_type.fieldNames():
                error_state.absent.add(f'{column}[].{name}')

    @staticmethod
    def parse_array_type(schema) -> ArrayType:
        if schema.inner_schema is None:
            return ArrayType(StringType())
        if schema.dtype in {'array', 'list'}:
            raise NotImplementedError('nested arrays are not supported yet')
        return ArrayType(ObjectTypeRule.parse_struct_type(schema.inner_schema))
