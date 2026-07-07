from typing import Callable, Optional, Union

from pyspark.sql import Column
from pyspark.sql import functions as fn
from pyspark.sql.types import ArrayType, DataType, StructField, StructType


def dtype_at(dtype: DataType, path: str) -> DataType:
    """
    Navigate a data type tree by a relative path (e.g. "b.c", "b[].c").

    :param dtype: the data type to start navigation from
    :param path: relative path, where parts ending with "[]" descend into array element types
    :return: DataType at the given path
    """

    if not path:
        return dtype

    for part in path.split('.'):
        if not isinstance(dtype, StructType):
            raise ValueError(f'Path "{path}" is not a nested column in the data type.')
        if '[]' in part:
            dtype = dtype[part[:-2]].dataType.elementType
        else:
            dtype = dtype[part].dataType

    return dtype


def dtype_set(dtype: Optional[DataType], path: str, new_dtype: DataType) -> DataType:
    """
    Return a copy of the data type tree with the type at the nested path replaced (creating missing fields).

    :param dtype: the data type to update (None or non-struct types are treated as an empty struct)
    :param path: relative path, where parts ending with "[]" descend into array element types
    :param new_dtype: the data type to set at the given path
    :return: updated data type tree
    """

    if not path:
        return new_dtype

    part, _, rest = path.partition('.')
    is_array = part.endswith('[]')
    name = part[:-2] if is_array else part

    existing = None
    if isinstance(dtype, StructType) and name in dtype.fieldNames():
        existing = dtype[name].dataType

    if is_array:
        elem = existing.elementType if isinstance(existing, ArrayType) else None
        field_type = ArrayType(dtype_set(elem, rest, new_dtype) if rest else new_dtype)
    else:
        field_type = dtype_set(existing, rest, new_dtype) if rest else new_dtype

    fields = list(dtype.fields) if isinstance(dtype, StructType) else []
    for i, field in enumerate(fields):
        if field.name == name:
            fields[i] = StructField(name, field_type, field.nullable, field.metadata)
            break
    else:
        fields.append(StructField(name, field_type, nullable=True, metadata={}))

    return StructType(fields)


def dtype_drop(dtype: DataType, path: str) -> DataType:
    """
    Return a copy of the data type tree with the field at the nested path removed.

    :param dtype: the data type to update
    :param path: relative path, where parts ending with "[]" descend into array element types
    :return: updated data type tree
    """

    part, _, rest = path.partition('.')
    is_array = part.endswith('[]')
    name = part[:-2] if is_array else part

    fields = list(dtype.fields)
    if not rest:
        return StructType([field for field in fields if field.name != name])

    for i, field in enumerate(fields):
        if field.name != name:
            continue
        if is_array:
            inner = field.dataType
            new_elem = dtype_drop(inner.elementType, rest)
            fields[i] = StructField(name, ArrayType(new_elem, inner.containsNull), field.nullable, field.metadata)
        else:
            fields[i] = StructField(name, dtype_drop(field.dataType, rest), field.nullable, field.metadata)

    return StructType(fields)


def nested_get_expr(column: str, root_expr: Optional[Column] = None) -> Column:
    """
    Expression-level counterpart of fn.col: return the value expression of a nested column
    navigated from the given root expression, flattening array element values with fn.transform.

    :param column: column name, which can be nested (e.g. "a.b.c", "a.b[].c")
    :param root_expr: current expression for the root column (falls back to fn.col of the root)
    :return: value expression for the column
    """

    def field_expr(path):
        parts = path.split('.')
        expr = root_expr if root_expr is not None else fn.col(parts[0])
        for part in parts[1:]:
            expr = expr.getField(part)
        return expr

    def get_nested(x):
        for part in nested[0].split('.'):
            x = x[part]
        return x

    if '[]' not in column:
        return field_expr(column)

    if column.endswith('[]'):
        return field_expr(column[:-2])

    root, *nested = column.split('[].')
    return fn.transform(field_expr(root), get_nested)


def nested_set_expr(
        col_name: str, val: Union[Column, Callable[[Column], Column]],
        root_expr: Optional[Column], root_dtype: Optional[DataType]
) -> Column:
    """
    Return the new value expression for the root column with the nested column set, creating a
    nested structure as needed and driven by a data type tree instead of a DataFrame schema.

    :param col_name: column name, which can be nested (e.g. "a.b.c")
    :param val: value to be assigned to the column
    :param root_expr: current expression for the root column (None if the column does not exist)
    :param root_dtype: current data type of the root column (None if the column does not exist)
    :return: new value expression for the root column
    """

    root, *nested_names = col_name.split('.')
    if is_array_root := root.endswith('[]'):
        root = root[:-2]

    exists = root_dtype is not None
    base = root_expr if root_expr is not None else fn.col(root)

    if not nested_names:
        if is_array_root:
            if not exists:
                return fn.array()
            fn_val = val
            if isinstance(val, Column):
                fn_val = lambda x: val  # pylint: disable=unnecessary-lambda-assignment
            return fn.transform(base, fn_val)
        return val

    def build_struct(fields, nested, dtype, col):
        is_array = nested[0].endswith('[]')

        struct_cols = []
        for field in fields:
            if field != nested[0].rstrip('[]'):
                struct_cols.append(col.getField(field).alias(field))
                continue

            if len(nested) > 1:
                field_dtype = dtype_at(dtype, nested[0])
                nested_fields = field_dtype.names

                if is_array:
                    def build_array(x):
                        # pylint: disable=cell-var-from-loop
                        return build_struct(nested_fields, nested[1:], field_dtype, x)

                    expr = fn.transform(col.getField(field), build_array)
                else:
                    expr = build_struct(nested_fields, nested[1:], field_dtype, col.getField(field))
                struct_cols.append(expr.alias(field))
            else:
                expr = val
                if is_array:
                    fn_val = val
                    if isinstance(val, Column):
                        fn_val = lambda x: val  # pylint: disable=unnecessary-lambda-assignment
                    expr = fn.transform(col.getField(field), fn_val).alias(field)
                else:
                    if not isinstance(val, Column):
                        expr = val(col.getField(field))

                struct_cols.append(expr.alias(field))

        if nested[0].rstrip('[]') not in fields:
            expr = val
            if is_array:
                expr = fn.array()
            elif len(nested) > 1:
                expr = build_struct([], nested[1:], None, col.getField(nested[0]))

            struct_cols.append(expr.alias(nested[0]))

        return fn.struct(*struct_cols)

    if root_dtype is not None and root_dtype.typeName() == 'struct':
        field_names, level_dtype = root_dtype.names, root_dtype
    elif root_dtype is not None and root_dtype.typeName() == 'array' and root_dtype.elementType.typeName() == 'struct':
        field_names, level_dtype = root_dtype.elementType.names, root_dtype.elementType
    else:
        field_names, level_dtype = [], None

    if is_array_root:
        if not exists:
            return fn.array()
        return fn.transform(base, lambda x: build_struct(field_names, nested_names, level_dtype, x))
    return build_struct(field_names, nested_names, level_dtype, base)


def nested_drop_expr(column: str, root_expr: Column, root_dtype: DataType) -> Column:
    """
    Return the new value expression for the root column with the nested column removed.

    :param column: column name, which can be nested (e.g. "a.b", "a[].b", "a.b[].c.d")
    :param root_expr: current expression for the root column
    :param root_dtype: current data type of the root column
    :return: new value expression for the root column
    """

    if '[].' in column:
        # drop a field from array element structs
        root, *nested = column.split('[].')

        def drop(x, nodes):
            if len(nodes) <= 1:
                return x.dropFields(nodes[0])

            return x.withField(nodes[0], drop(x[nodes[0]], nodes[1:]))

        return nested_set_expr(root + '[]', lambda x: drop(x, nested[0].split('.')), root_expr, root_dtype)

    # rebuild parent structs skipping the dropped field
    _, *nested_names = column.split('.')

    def build_struct(fields, nested, dtype, col):
        struct_cols = []
        for field in fields:
            if field != nested[0]:
                struct_cols.append(col.getField(field).alias(field))
                continue

            if len(nested) > 1:
                field_dtype = dtype_at(dtype, field)
                struct_cols.append(
                    build_struct(field_dtype.names, nested[1:], field_dtype, col.getField(field)).alias(field))

        return fn.struct(*struct_cols)

    return build_struct(root_dtype.names, nested_names, root_dtype, root_expr)
