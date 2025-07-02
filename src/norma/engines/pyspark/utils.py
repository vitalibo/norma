import random
import string
from typing import Callable

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import DataType, StructType


def backup_col(column, error_state):
    """
    Build a backup column name for a given column
    """

    return f'{suffix_col(column, error_state)}_bak'


def suffix_col(column, error_state):
    """
    Build a suffix for a given column.
    """

    if column in error_state.suffixes:
        return error_state.suffixes[column]

    while True:
        suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        if suffix not in error_state.suffixes.values():
            error_state.suffixes[column] = suffix
            return suffix


def data_type_of(df: DataFrame, column: str) -> DataType:
    """
    Get a data type of column in a DataFrame including nested columns.

    :param df: DataFrame
    :param column: Column name, which can be nested (e.g. "a.b.c")
    :return: DataType of the column
    """

    if '.' in df.schema or column in df.schema:
        return df.schema[column].dataType

    parts = column.split('.')
    struct_field = df.schema[parts[0]]
    for part in parts[1:]:
        if isinstance(struct_field.dataType, StructType):
            struct_field = struct_field.dataType[part]
        else:
            raise ValueError(f'Column "{column}" is not a nested column in the DataFrame schema.')

    return struct_field.dataType


def with_nested_column_renamed(existing: str, new: str) -> Callable[[DataFrame], DataFrame]:
    """
    Rename a nested column in a DataFrame.

    :param existing: existing column name, which can be nested (e.g. "a.b.c")
    :param new: new column name
    :return: function that can be used to transform a DataFrame
    """

    def transform(df):
        root, *nested_names = existing.split('.')
        if not nested_names:
            return df.withColumnRenamed(existing, new)

        def build_struct(fields, nested, path):
            struct_cols = []
            for field in fields:
                if field != nested[0]:
                    struct_cols.append(fn.col(f'{path}.{field}').alias(field))
                    continue

                if len(nested) > 1:
                    nested_field = data_type_of(df, f'{path}.{field}').names
                    struct_cols.append(build_struct(nested_field, nested[1:], f'{path}.{field}').alias(field))

            return fn.struct(*struct_cols)

        return df \
            .withColumn(new, fn.col(existing)) \
            .withColumn(root, build_struct(df.schema[root].dataType.names, nested_names, root))

    return transform


def with_nested_column(col_name: str, col: Column) -> Callable[[DataFrame], DataFrame]:
    """
    Create a new column in a DataFrame with a nested structure.

    :param col_name: column name, which can be nested (e.g. "a.b.c")
    :param col: value to be assigned to the column
    :return: function that can be used to transform a DataFrame
    """

    def transform(df):
        root, *nested_names = col_name.split('.')
        if not nested_names:
            return df.withColumn(root, col)

        try:
            field_names = df.schema[root].dataType.names
        except KeyError:
            return df.withColumn(root, fn.struct(col.alias(nested_names[0])))

        def build_struct(fields, nested, path):
            struct_cols = []
            for field in fields:
                if field != nested[0]:
                    struct_cols.append(fn.col(f'{path}.{field}').alias(field))
                    continue

                if len(nested) > 1:
                    nested_field = data_type_of(df, f'{path}.{field}').names
                    struct_cols.append(build_struct(nested_field, nested[1:], f'{path}.{field}').alias(field))
                else:
                    struct_cols.append(col.alias(field))

            if nested[0] not in fields:
                expr = fn.lit(None)
                if len(nested) > 1:
                    expr = build_struct([], nested[1:], f'{path}.{nested[0]}')

                struct_cols.append(expr.alias(nested[0]))

            return fn.struct(*struct_cols)

        return df.withColumn(root, build_struct(field_names, nested_names, root))

    return transform


def default_if_null(default: Column):
    """
    Create a function that returns a column with a default value if the original column is null.
    """

    def wrap(col):
        return fn.coalesce(col, default)

    return wrap
