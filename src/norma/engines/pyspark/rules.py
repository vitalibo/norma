import abc
import inspect
import json
from functools import partial, reduce
from typing import Any, Iterable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import (
    BooleanType,
    DateType,
    FloatType,
    IntegerType,
    MapType,
    NumericType,
    StringType,
    StructField,
    StructType,
    TimestampType
)

from norma import errors
from norma.engines.pyspark.utils import (
    backup_col,
    data_type_of,
    suffix_col,
    with_nested_column,
    with_nested_column_renamed
)
from norma.rules import ErrorState as IErrorState
from norma.rules import Rule


class ErrorState(IErrorState):
    """
    Error state for PySpark DataFrame validation

    :param error_column: The name of the column to store error information
    """

    def __init__(self, error_column: str):
        self.error_column = error_column
        self.suffixes = {}

    def add_errors(self, boolmask: Column, column: str, **kwargs):
        """
        Returns a function that needs to be applied to a DataFrame to add errors to the error state
        """

        details = kwargs.get('details') or kwargs
        details_col = fn.when(boolmask, fn.struct(*[fn.lit(v).alias(k) for k, v in details.items()]))

        name = f'{self.error_column}_{suffix_col(column, self)}'
        return lambda df: df.withColumn(name, fn.array_append(fn.col(name), details_col))


class BaseRule(Rule):
    """
    Base rule class for PySpark DataFrame validation

    :param func: The function to apply to the DataFrame
    :param kwargs: Additional keyword arguments to pass to the function
    """

    def __init__(self, func, **kwargs):
        self.func = func
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
            if 'error_state' in signature.parameters:
                params['error_state'] = error_state
            return params

        func_params = inspect_params(self.func)
        if 'df' in func_params:
            return self.func(**func_params)

        if '__pre_func__' in self.kwargs:
            pre_func = self.kwargs['__pre_func__']
            df = pre_func(**inspect_params(pre_func))

        return df.transform(error_state.add_errors(self.func(**func_params), column, **self.kwargs))


def rule(func, **kwargs) -> BaseRule:
    return BaseRule(func, **kwargs)


def required() -> Rule:
    @Rule.new
    def verify(df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        mask = fn.isnull(fn.col(column))
        try:
            data_type_of(df, column)
        except Exception:  # pylint: disable=broad-except
            df = df.transform(with_nested_column(column, fn.lit(None)))
            mask = fn.lit(True)

        return df.transform(error_state.add_errors(mask, column, details=errors.MISSING))

    return verify


def equal_to(eq: Any) -> Rule:
    return rule(
        lambda col: fn.col(col) != fn.lit(eq),
        details=errors.EQUAL_TO.format(eq=eq)
    )


def not_equal_to(ne: Any) -> Rule:
    return rule(
        lambda col: fn.col(col) == fn.lit(ne),
        details=errors.NOT_EQUAL_TO.format(ne=ne)
    )


def greater_than(gt: Any) -> Rule:
    return rule(
        lambda col: fn.col(col) <= fn.lit(gt),
        details=errors.GREATER_THAN.format(gt=gt)
    )


def greater_than_equal(ge: Any) -> Rule:
    return rule(
        lambda col: fn.col(col) < fn.lit(ge),
        details=errors.GREATER_THAN_EQUAL.format(ge=ge)
    )


def less_than(lt: Any) -> Rule:
    return rule(
        lambda col: fn.col(col) >= fn.lit(lt),
        details=errors.LESS_THAN.format(lt=lt)
    )


def less_than_equal(le: Any) -> Rule:
    return rule(
        lambda col: fn.col(col) > fn.lit(le),
        details=errors.LESS_THAN_EQUAL.format(le=le)
    )


def multiple_of(multiple: Any) -> Rule:
    def before(df, col):
        if not isinstance(data_type_of(df, col), NumericType):
            raise ValueError('multiple_of rule can only be applied to numeric columns')
        return df

    return rule(
        lambda col: (fn.col(col) % fn.lit(multiple)) != fn.lit(0),
        details=errors.MULTIPLE_OF.format(multiple_of=multiple),
        __pre_func__=before
    )


def min_length(value: int) -> Rule:
    def before(df, col):
        if not isinstance(data_type_of(df, col), StringType):
            raise ValueError('min_length rule can only be applied to string columns')
        return df

    return rule(
        lambda col: fn.length(fn.col(col)) < value,
        details=errors.STRING_TOO_SHORT.format(min_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before
    )


def max_length(value: int) -> Rule:
    def before(df, col):
        if not isinstance(data_type_of(df, col), StringType):
            raise ValueError('max_length rule can only be applied to string columns')
        return df

    return rule(
        lambda col: fn.length(fn.col(col)) > value,
        details=errors.STRING_TOO_LONG.format(max_length=value, _plural_='s' if value > 1 else ''),
        __pre_func__=before
    )


def pattern(regex: str) -> Rule:
    def before(df, col):
        if not isinstance(data_type_of(df, col), StringType):
            raise ValueError('pattern rule can only be applied to string columns')
        return df

    return rule(
        lambda col: ~fn.col(col).rlike(regex),
        details=errors.STRING_PATTERN_MISMATCH.format(pattern=regex),
        __pre_func__=before
    )


def isin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda col: ~fn.col(col).isin(values),
        details=errors.ENUM.format(expected=values)
    )


def notin(values: Iterable[Any]) -> Rule:
    return rule(
        lambda col: fn.col(col).isin(values),
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
    return NumericTypeRule(IntegerType, errors.INT_TYPE, errors.INT_PARSING)


def float_parsing():
    return NumericTypeRule(FloatType, errors.FLOAT_TYPE, errors.FLOAT_PARSING)


def str_parsing() -> Rule:
    return StringTypeRule()


def bool_parsing() -> Rule:
    return BooleanTypeRule()


def datetime_parsing() -> Rule:
    return DatetimeTypeRule()


def date_parsing() -> Rule:
    return DateTypeRule()


def object_parsing(schema) -> Rule:
    return ObjectTypeRule(schema)


class DataTypeRule(Rule):
    """
    Abstract base class for data type rules
    """

    def __init__(self, dtype, supported, details):
        self.dtype = dtype
        self.supported = supported
        self.details = details

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        data_type = data_type_of(df, column)
        if isinstance(data_type, self.dtype):
            return df

        if data_type.typeName() == 'void':
            return df.withColumn(column, fn.col(column).cast(self.dtype.typeName()))

        backup_column = backup_col(column, error_state)
        df = df.withColumn(backup_column, fn.col(column))
        if not isinstance(data_type, self.supported):
            return df \
                .withColumn(column, fn.lit(None).cast(self.dtype.typeName())) \
                .transform(error_state.add_errors(fn.lit(True), column, details=self.details))

        return self.cast(df, column, fn.col(backup_column), error_state)

    @abc.abstractmethod
    def cast(self, df: DataFrame, column: str, backup_column: Column, error_state: ErrorState) -> DataFrame:
        """
        Abstract method to cast the column to the specified data type
        """


class NumericTypeRule(DataTypeRule):
    """
    Class for numeric type casting rules
    """

    def __init__(self, dtype, numeric_type, numeric_parsing):
        super().__init__(dtype, (StringType, NumericType, BooleanType), numeric_type)
        self.numeric_parsing = numeric_parsing

    def cast(self, df: DataFrame, column: str, backup_column: Column, error_state: ErrorState) -> DataFrame:
        return df \
            .transform(with_nested_column(column, fn.col(column).cast(self.dtype.typeName()))) \
            .transform(error_state.add_errors(fn.isnull(fn.col(column)) & fn.isnotnull(backup_column), column,
                                              details=self.numeric_parsing))


class StringTypeRule(DataTypeRule):
    """
    Class for string type casting rules
    """

    def __init__(self):
        super().__init__(StringType, (NumericType, BooleanType, DateType, TimestampType), errors.STRING_TYPE)

    def cast(self, df: DataFrame, column: str, backup_column: Column, error_state: ErrorState) -> DataFrame:
        return df.transform(with_nested_column(column, fn.col(column).cast('string')))


class BooleanTypeRule(DataTypeRule):
    """
    Class for boolean type casting rules
    """

    def __init__(self):
        super().__init__(BooleanType, (NumericType, StringType), errors.BOOL_TYPE)

    def cast(self, df: DataFrame, column: str, backup_column: Column, error_state: ErrorState) -> DataFrame:
        expr = fn.col(column)
        if isinstance(data_type_of(df, column), StringType):
            expr = fn.lower(fn.trim(fn.col(column)))
            expr = fn.when(expr.isin(['on', 'off']), expr == 'on').otherwise(fn.col(column).cast('boolean'))

        return df \
            .transform(with_nested_column(column, expr.cast('boolean'))) \
            .transform(error_state.add_errors(fn.isnull(fn.col(column)) & fn.isnotnull(backup_column), column,
                                              details=errors.BOOL_PARSING))


class DatetimeTypeRule(DataTypeRule):
    """
    Class for datetime type casting rules
    """

    def __init__(self):
        super().__init__(TimestampType, (StringType, DateType), errors.DATETIME_TYPE)

    def cast(self, df: DataFrame, column: str, backup_column: Column, error_state: ErrorState) -> DataFrame:
        return df \
            .transform(with_nested_column(column, fn.to_timestamp(fn.col(column)))) \
            .transform(error_state.add_errors(fn.isnull(fn.col(column)) & fn.isnotnull(backup_column), column,
                                              details=errors.DATETIME_PARSING))


class DateTypeRule(DataTypeRule):
    """
    Class for date type casting rules
    """

    def __init__(self):
        super().__init__(DateType, (StringType, TimestampType), errors.DATE_TYPE)

    def cast(self, df: DataFrame, column: str, backup_column: Column, error_state: ErrorState) -> DataFrame:
        return df \
            .transform(with_nested_column(column, fn.to_date(fn.col(column)))) \
            .transform(error_state.add_errors(fn.isnull(fn.col(column)) & fn.isnotnull(backup_column), column,
                                              details=errors.DATE_PARSING))


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
            return df \
                .transform(with_nested_column(column, new_struct))

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
