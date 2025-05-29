import inspect
import json
from functools import partial, reduce
from typing import Any, Iterable, Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import (
    ArrayType, BooleanType, DateType, FloatType, IntegerType, MapType, NumericType, StringType, StructField, StructType,
    TimestampType
)

from norma import errors
from norma.engines.pyspark.utils import (
    backup_col, data_type_of, suffix_col, with_nested_column, with_nested_column_renamed
)
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    Error state for PySpark DataFrame validation

    :param error_column: The name of the column to store error information
    """

    def __init__(self, error_column: str, has_array: bool):
        self.error_column = error_column
        self.has_array = has_array
        self.suffixes = {}

    def add_errors(self, boolmask: Column, column: str, **kwargs):
        def array_strategy(df):
            cols = [fn.lit(v).alias(k) for k, v in details.items()]

            indexes_col = f'{suffix}_indexes'
            if indexes_col in df.columns:
                df = df.withColumn(indexes_col, fn.zip_with(fn.col(indexes_col), boolmask, lambda x, y: x | y))
            else:
                df = df.withColumn(indexes_col, boolmask)

            indexes = fn.filter(
                fn.transform(boolmask, lambda x, i: fn.when(x, i).otherwise(fn.lit(None))), lambda x: x.isNotNull())
            cols.append(indexes.alias('loc'))

            details_col = fn.when(fn.array_size(indexes) > 0, fn.struct(*cols))
            error_column = f'{self.error_column}_{suffix_col(column + "[]", self)}'
            if error_column not in df.columns:
                df = df.withColumn(error_column, fn.array())
            return df.withColumn(error_column, fn.array_append(fn.col(error_column), details_col))

        def default_strategy(df):
            cols = [fn.lit(v).alias(k) for k, v in details.items()]
            # if DataFrame has at least one array column, we need to add indexes
            # because we cannot append a struct to an array with different types
            if self.has_array:
                cols.append(fn.lit(None).cast('array<int>').alias('loc'))

            details_col = fn.when(boolmask, fn.struct(*cols))
            error_column = f'{self.error_column}_{suffix}'
            return df.withColumn(error_column, fn.array_append(fn.col(error_column), details_col))

        def func(df):
            # hack to infer the data type of the column,
            # and based on that choose the right function
            tmp_df = df.withColumn(f'{suffix}_tmp', boolmask)
            tmp_data_type = data_type_of(tmp_df, f'{suffix}_tmp')
            if tmp_data_type.simpleString() == 'array<boolean>':
                return array_strategy(df)
            return default_strategy(df)

        suffix = suffix_col(column, self)
        details = dict(kwargs.get('details'))
        return func


class BaseRule(Rule):
    """
    Base rule class for PySpark DataFrame validation

    :param func: The function to apply to the DataFrame
    :param kwargs: Additional keyword arguments to pass to the function
    """

    def __init__(self, func, details=None, **kwargs):
        self.func = func
        self.details = details
        self.kwargs = kwargs

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
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
            if set(signature.parameters) & {'col_expr', 'column_expr'}:
                params['col_expr' if 'col_expr' in signature.parameters else 'column_expr'] = fn.col(column)
            if 'error_state' in signature.parameters:
                params['error_state'] = error_state
            return params

        func_params = inspect_params(self.func)
        if 'df' in func_params:
            return self.func(**func_params)

        if '__pre_func__' in self.kwargs:
            pre_func = self.kwargs['__pre_func__']
            df = pre_func(**inspect_params(pre_func))

        if not self.__dict__.get('array', False):
            return df.transform(error_state.add_errors(self.func(**func_params), column, details=self.details))

        indexes = fn.transform(fn.col(column), self.func)
        return df.transform(error_state.add_errors(indexes, column, details=self.details))


def rule(func, **kwargs) -> BaseRule:
    return BaseRule(func, **kwargs)


def required() -> Rule:
    class NewRule(BaseRule):
        """
        Rule to check if a column is required (not null)
        """

        def __init__(self):
            super().__init__(
                lambda col_expr: fn.isnull(col_expr),  # pylint: disable=unnecessary-lambda
                details=errors.MISSING
            )

        def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
            if self.__dict__.get('array', False):
                return super().verify(df, column, error_state)

            mask = fn.isnull(fn.col(column))
            try:
                data_type_of(df, column)
            except Exception:  # pylint: disable=broad-except
                df = df.transform(with_nested_column(column, fn.lit(None)))
                mask = fn.lit(True)

            return df.transform(error_state.add_errors(mask, column, details=errors.MISSING))

    return NewRule()


def equal_to(eq: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr != fn.lit(eq),
        details=errors.EQUAL_TO.format(eq=eq)
    )


def not_equal_to(ne: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr == fn.lit(ne),
        details=errors.NOT_EQUAL_TO.format(ne=ne)
    )


def greater_than(gt: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr <= fn.lit(gt),
        details=errors.GREATER_THAN.format(gt=gt)
    )


def greater_than_equal(ge: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr < fn.lit(ge),
        details=errors.GREATER_THAN_EQUAL.format(ge=ge)
    )


def less_than(lt: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr >= fn.lit(lt),
        details=errors.LESS_THAN.format(lt=lt)
    )


def less_than_equal(le: Any) -> Rule:
    return rule(
        lambda col_expr: col_expr > fn.lit(le),
        details=errors.LESS_THAN_EQUAL.format(le=le)
    )


def multiple_of(multiple: Any) -> Rule:
    def before(df, col):
        data_type = data_type_of(df, col)
        if isinstance(data_type, ArrayType):
            if not isinstance(data_type.elementType, NumericType):
                raise ValueError('multiple_of rule can only be applied to numeric columns')
        elif not isinstance(data_type, NumericType):
            raise ValueError('multiple_of rule can only be applied to numeric columns')
        return df

    return rule(
        lambda col_expr: (col_expr % fn.lit(multiple)) != fn.lit(0),
        details=errors.MULTIPLE_OF.format(multiple_of=multiple),
        __pre_func__=before
    )


def min_length(value: int) -> Rule:
    def before(df, col):
        data_type = data_type_of(df, col)
        if isinstance(data_type, ArrayType):
            if not isinstance(data_type.elementType, StringType):
                raise ValueError('min_length rule can only be applied to string columns')
        elif not isinstance(data_type, StringType):
            raise ValueError('min_length rule can only be applied to string columns')
        return df

    return SequenceRule(
        lambda col_expr: fn.length(col_expr) < value,
        errors.STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else ''),
        array_func=lambda col_expr: fn.array_size(col_expr) < value,
        array_details=errors.TOO_SHORT.format(_type_='Array', min_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before
    )


def max_length(value: int) -> Rule:
    def before(df, col):
        data_type = data_type_of(df, col)
        if isinstance(data_type, ArrayType):
            if not isinstance(data_type.elementType, StringType):
                raise ValueError('max_length rule can only be applied to string columns')
        elif not isinstance(data_type, StringType):
            raise ValueError('max_length rule can only be applied to string columns')
        return df

    return SequenceRule(
        lambda col_expr: fn.length(col_expr) > value,
        errors.STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else ''),
        array_func=lambda col_expr: fn.array_size(col_expr) > value,
        array_details=errors.TOO_LONG.format(_type_='Array', max_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before
    )


def pattern(regex: str) -> Rule:
    def before(df, col):
        data_type = data_type_of(df, col)
        if isinstance(data_type, ArrayType):
            if not isinstance(data_type.elementType, StringType):
                raise ValueError('pattern rule can only be applied to string columns')
        elif not isinstance(data_type, StringType):
            raise ValueError('pattern rule can only be applied to string columns')
        return df

    return rule(
        lambda col_expr: ~col_expr.rlike(regex),
        details=errors.STRING_PATTERN_MISMATCH.format(pattern=regex),
        __pre_func__=before
    )


def isin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda col_expr: ~col_expr.isin(values),
        details=errors.ENUM.format(expected=values)
    )


def notin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda col_expr: col_expr.isin(values),
        details=errors.NOT_ENUM.format(unexpected=values)
    )


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    @Rule.new
    def verify(df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        if column in allowed:
            return df

        return df \
            .transform(with_nested_column_renamed(column, backup_col(column, error_state))) \
            .transform(error_state.add_errors(fn.lit(True), column, details=errors.EXTRA_FORBIDDEN))

    return verify


def int_parsing() -> Rule:
    return DataTypeRule(
        lambda col: col.cast('integer'),
        IntegerType, (StringType, NumericType, BooleanType), errors.INT_TYPE, errors.INT_PARSING
    )


def float_parsing():
    return DataTypeRule(
        lambda col: col.cast('float'),
        FloatType, (StringType, NumericType, BooleanType), errors.FLOAT_TYPE, errors.FLOAT_PARSING
    )


def str_parsing() -> Rule:
    return DataTypeRule(
        lambda col: col.cast('string'),
        StringType, (NumericType, BooleanType, DateType, TimestampType), errors.STRING_TYPE
    )


def bool_parsing() -> Rule:
    return BooleanTypeRule(
        lambda col: col.cast('boolean'),
        BooleanType, (NumericType, StringType), errors.BOOL_TYPE, errors.BOOL_PARSING
    )


def datetime_parsing() -> Rule:
    return DataTypeRule(
        lambda col: fn.to_timestamp(col),  # pylint: disable=unnecessary-lambda
        TimestampType, (StringType, DateType), errors.DATETIME_TYPE, errors.DATETIME_PARSING
    )


def date_parsing() -> Rule:
    return DataTypeRule(
        lambda col: fn.to_date(col),  # pylint: disable=unnecessary-lambda
        DateType, (StringType, TimestampType), errors.DATE_TYPE, errors.DATE_PARSING
    )


def uuid_parsing() -> Rule:
    uuid_regex = '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    return DataTypeRule(
        lambda col: fn.when(fn.lower(col).rlike(uuid_regex), fn.lower(col)),
        DateType, (StringType,), errors.UUID_TYPE, errors.UUID_PARSING
    )


def object_parsing(schema) -> Rule:
    return ObjectTypeRule(schema)


def array_parsing(schema) -> Rule:
    return ArrayTypeRule(schema)


class DataTypeRule(Rule):
    """
    Abstract base class for data type rules
    """

    def __init__(  # pylint: disable=too-many-arguments
            self, cast, dtype, supported, type_details=None, parsing_details=None
    ):
        self.cast = cast
        self.dtype = dtype
        self.supported = supported
        self.type_details = type_details or {}
        self.parsing_details = parsing_details or {}

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        if self.__dict__.get('array', False):
            return self._array_verify_strategy(df, column, error_state)
        return self._default_verify_strategy(df, column, error_state)

    def _default_verify_strategy(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        data_type = data_type_of(df, column)
        if isinstance(data_type, self.dtype):
            return df

        if data_type.typeName() == 'void':
            return df.withColumn(column, fn.col(column).cast(self.dtype.typeName()))

        backup_column = backup_col(column, error_state)
        df = df.withColumn(backup_column, fn.col(column))
        if not isinstance(data_type, self.supported):
            return df \
                .transform(with_nested_column(column, fn.lit(None).cast(self.dtype.typeName()))) \
                .transform(error_state.add_errors(fn.lit(True), column, details=self.type_details))

        df = df.transform(with_nested_column(column, self.cast(fn.col(column))))
        if not self.parsing_details:
            return df

        return df \
            .transform(error_state.add_errors(fn.isnull(fn.col(column)) & fn.isnotnull(backup_column), column,
                                              details=self.parsing_details))

    def _array_verify_strategy(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        data_type = data_type_of(df, column)
        element_type = data_type.elementType
        if isinstance(element_type, self.dtype):
            return df

        if element_type.typeName() == 'void':
            return df.withColumn(column, fn.col(column).cast(ArrayType(self.dtype())))

        backup_column = backup_col(column, error_state)
        df = df.withColumn(f'{backup_column}_array', fn.col(column))
        if not isinstance(element_type, self.supported):
            indexes = fn.transform(fn.col(column), lambda x: fn.lit(True))
            return df \
                .transform(with_nested_column(
                column, fn.transform(fn.col(column), lambda x: fn.lit(None).cast(self.dtype())))) \
                .transform(error_state.add_errors(indexes, column, details=self.type_details))

        df = df.transform(with_nested_column(column, fn.transform(fn.col(column), self.cast)))
        if not self.parsing_details:
            return df

        indexes = fn.zip_with(
            fn.col(column), fn.col(f'{backup_column}_array'), lambda x, y: fn.isnull(x) & fn.isnotnull(y))
        return df.transform(error_state.add_errors(indexes, column, details=self.parsing_details))


class BooleanTypeRule(DataTypeRule):
    """
    Class for boolean type casting rules
    """

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        # special case for string columns where we want to cast 'on'/'off' to boolean
        def cast_str_as_bool(col):
            expr = fn.lower(fn.trim(col))
            expr = fn.when(expr.isin(['on', 'off']), expr == 'on').otherwise(col.cast('boolean'))
            return expr.cast('boolean')

        data_type = data_type_of(df, column)
        if (isinstance(data_type, StringType)
                or (isinstance(data_type, ArrayType) and isinstance(data_type.elementType, StringType))):
            self.cast = cast_str_as_bool

        return super().verify(df, column, error_state)


class ObjectTypeRule(Rule):
    """
    Class for object type casting rules
    """

    def __init__(self, schema):
        self.struct_type = self.parse_struct_type(schema)

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        data_type = data_type_of(df, column)
        if isinstance(data_type, StructType):
            return df

        if data_type.typeName() == 'void':
            return df.withColumn(column, fn.col(column).cast(self.struct_type))

        backup_column = backup_col(column, error_state)
        df = df.withColumn(backup_column, fn.col(column))
        if not isinstance(data_type, (StringType, MapType)):
            return df \
                .transform(with_nested_column(column, fn.lit(None).cast(self.struct_type))) \
                .transform(error_state.add_errors(fn.lit(True), column, details=errors.OBJECT_TYPE))

        if isinstance(data_type, MapType):
            new_struct = fn.struct(*(fn.col(column)[field].alias(field) for field in self.struct_type.fieldNames()))
            return df.transform(with_nested_column(column, new_struct))

        return self._cast_json_str(df, column, fn.col(backup_column), error_state)

    def _cast_json_str(self, df: DataFrame, column: str, backup_column: Column, error_state: ErrorState) -> DataFrame:
        @fn.udf(returnType=BooleanType())
        def is_malformed(struct_fields_is_null_and_origin_is_not_null, val):
            if not struct_fields_is_null_and_origin_is_not_null:
                return False

            try:
                json.loads(val)
                return False
            except:  # pylint: disable=bare-except
                return True

        is_malformed = is_malformed(
            reduce(
                lambda a, b: a & b,
                (fn.isnull(fn.col(f'{column}.{field}')) for field in self.struct_type.fieldNames())) &
            fn.isnotnull(backup_column), backup_column)

        return df \
            .withColumn(column, fn.from_json(fn.col(column), self.struct_type)) \
            .transform(with_nested_column(column, fn.when(~is_malformed, fn.col(column)))) \
            .transform(error_state.add_errors(fn.isnull(fn.col(column)) & fn.isnotnull(backup_column), column,
                                              details=errors.OBJECT_PARSING))

    @staticmethod
    def parse_struct_type(schema) -> StructType:
        def struct_field(name, col):
            return StructField(
                name,
                {
                    'object': partial(ObjectTypeRule.parse_struct_type, col.inner_schema),
                    'array': partial(ArrayTypeRule.parse_array_type, col.inner_schema),
                    'str': StringType,
                    'string': StringType,
                    'int': IntegerType,
                    'integer': IntegerType,
                    'float': FloatType,
                    'double': FloatType,
                    'number': FloatType,
                    'bool': BooleanType,
                    'boolean': BooleanType,
                    'datetime': TimestampType,
                    'date': DateType,
                }[col.dtype](),
                nullable=True,
                metadata={}
            )

        return StructType([struct_field(k, v) for k, v in schema.columns.items()])


class ArrayTypeRule(Rule):
    """
    Class for array type casting rules
    """

    def __init__(self, schema):
        self.struct_type = self.parse_array_type(schema)

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        data_type = data_type_of(df, column)
        if isinstance(data_type, ArrayType):
            return df

        if data_type.typeName() == 'void':
            return df.withColumn(column, fn.col(column).cast(self.struct_type))

        backup_column = backup_col(column, error_state)
        df = df.withColumn(backup_column, fn.col(column))
        if not isinstance(data_type, (StringType, ArrayType)):
            return df \
                .transform(with_nested_column(column, fn.lit(None).cast(self.struct_type))) \
                .transform(error_state.add_errors(fn.lit(True), column, details=errors.ARRAY_TYPE))

        return df \
            .transform(with_nested_column(column, fn.from_json(fn.col(column), self.struct_type))) \
            .transform(error_state.add_errors(fn.isnull(fn.col(column)) & fn.isnotnull(fn.col(backup_column)), column,
                                              details=errors.ARRAY_PARSING))

    @staticmethod
    def parse_array_type(schema) -> ArrayType:  # pylint: disable=unused-argument
        # TODO: support nested structs in arrays
        return ArrayType(StringType())


class SequenceRule(BaseRule):
    """
    Rule to apply different functions based on the data type of the column.
    """

    def __init__(self, func, details, array_func, array_details, **kwargs):
        super().__init__(func, details=details, **kwargs)
        self.array_func = array_func
        self.array_details = array_details

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> Optional[DataFrame]:
        data_type = data_type_of(df, column)
        if isinstance(data_type, ArrayType) and not self.__dict__.get('array', False):
            self.func = self.array_func
            self.details = self.array_details

        return super().verify(df, column, error_state)
