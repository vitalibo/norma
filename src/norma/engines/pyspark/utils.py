from __future__ import annotations

import functools
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
        dtype = dtype[part.removesuffix('[]')].dataType
        if '[]' in part:
            dtype = dtype.elementType

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


class PendingStruct:
    """
    Pending value of a struct that keeps every assigned field apart from its siblings.
    """

    def __init__(self, base: Optional[Column]) -> None:
        self.base = base
        self.fields = {}

    def get(self, name: str) -> Union[Column, PendingStruct]:
        """
        Return the pending value of a direct field (an overlay for a struct field assigned deeper).
        """

        if name in self.fields:
            return self.fields[name]
        return self.base.getField(name)

    def materialize(self, dtype: Optional[DataType]) -> Column:
        """
        Materialize the struct with the fields of the given data type, in its order.
        """

        names = list(dtype.names) if isinstance(dtype, StructType) else []
        names += [name for name in self.fields if name not in names]
        struct = fn.struct(*(materialize(self.get(name), _field_dtype(dtype, name)).alias(name) for name in names))
        return struct if self.base is None else fn.when(self.base.isNotNull(), struct)

    def is_not_null(self) -> Column:
        """
        Return whether the materialized struct is not null, without building the struct itself.
        """

        return fn.lit(True) if self.base is None else self.base.isNotNull()


class PendingArrayStruct:
    """
    Pending value of an array of structs that keeps every assigned element field as its own array, aligned with the
    base array.
    """

    def __init__(self, base: Column) -> None:
        self.base = base
        self.fields = {}
        # set once a field is dropped: the elements must then be rebuilt even without any assigned field
        self.reshaped = False

    def get(self, name: str) -> Union[Column, PendingArrayStruct]:
        """
        Return the pending values of a direct element field, as an array aligned with the base array.
        """

        if name in self.fields:
            return self.fields[name]
        return fn.transform(self.base, lambda x: x.getField(name))

    def materialize(self, dtype: Optional[DataType]) -> Column:
        """
        Materialize the array with elements holding the fields of the given data type, in its order.
        """

        if not self.fields and not self.reshaped:
            return self.base

        element_dtype = dtype.elementType if isinstance(dtype, ArrayType) else None
        names = list(element_dtype.names) if isinstance(element_dtype, StructType) else []
        names += [name for name in self.fields if name not in names]
        slots = {name: f'_{i}' for i, name in enumerate((name for name in names if name in self.fields), start=1)}

        # the field arrays are zipped rather than indexed from a lambda, which would evaluate them once per element
        zipped = fn.arrays_zip(self.base.alias('_0'), *(
            materialize(self.fields[name], _array_of(_field_dtype(element_dtype, name))).alias(slot)
            for name, slot in slots.items()
        ))

        def element_of(item):
            struct = fn.struct(*(
                (item[slots[name]] if name in slots else item['_0'].getField(name)).alias(name) for name in names
            ))
            return fn.when(item['_0'].isNotNull(), struct)

        return fn.transform(zipped, element_of)

    def is_not_null(self) -> Column:
        """
        Return whether each materialized element is not null, without building the elements themselves.
        """

        return fn.transform(self.base, fn.isnotnull)


Pending = Union[Column, PendingStruct, PendingArrayStruct]


def materialize(value: Optional[Pending], dtype: Optional[DataType]) -> Optional[Column]:
    """
    Materialize a pending value into an expression.
    """

    return value.materialize(dtype) if isinstance(value, (PendingStruct, PendingArrayStruct)) else value


def _element_path_of(column: str) -> Optional[list]:
    """
    Return the path of a column below the root array elements (e.g. ["b", "c"] for "a[].b.c"), or None for a column
    that is not inside the root array elements or crosses another array.
    """

    _, sep, path = column.partition('[].')
    if not sep or '[]' in path:
        return None
    return path.split('.')


def _field_dtype(dtype: Optional[DataType], name: str) -> Optional[DataType]:
    if isinstance(dtype, StructType) and name in dtype.names:
        return dtype[name].dataType
    return None


def _element_dtype(dtype: Optional[DataType]) -> Optional[DataType]:
    return dtype.elementType if isinstance(dtype, ArrayType) else None


def _array_of(dtype: Optional[DataType]) -> Optional[DataType]:
    return None if dtype is None else ArrayType(dtype)


def _array_overlay_get(overlay: PendingArrayStruct, path: list, element_dtype: Optional[DataType]):
    """
    Return the pending values of a field below the elements of an array overlay, as an array aligned with it (or a
    nested overlay), along with the element data type of that array.
    """

    value, dtype = overlay, element_dtype
    for i, part in enumerate(path):
        if not isinstance(value, PendingArrayStruct):
            rest = path[i:]
            value = fn.transform(value, lambda x: functools.reduce(Column.getField, rest, x))  # noqa: B023
            for name in rest:
                dtype = _field_dtype(dtype, name)
            break
        value, dtype = value.get(part), _field_dtype(dtype, part)
    return value, dtype


def _array_overlay_at(root_expr, root, path: list, root_dtype: Optional[DataType]):
    """
    Return the array overlay holding the parent of a field below the root array elements, creating the overlays along
    the path, along with the root overlay and the element data type of the parent. Return None when a level above the
    field is not a struct.
    """

    dtype = _element_dtype(root_dtype)
    for part in path[:-1]:
        dtype = _field_dtype(dtype, part)
        if not isinstance(dtype, StructType):
            return None
    if not isinstance(_element_dtype(root_dtype), StructType):
        return None

    top = root_expr if isinstance(root_expr, PendingArrayStruct) else PendingArrayStruct(
        root_expr if root_expr is not None else fn.col(root)
    )
    overlay, dtype = top, _element_dtype(root_dtype)
    for part in path[:-1]:
        nested = overlay.fields.get(part)
        if not isinstance(nested, PendingArrayStruct):
            nested = overlay.fields[part] = PendingArrayStruct(materialize(overlay.get(part), None))
        overlay, dtype = nested, _field_dtype(dtype, part)
    return top, overlay, dtype


def _overlay_of(value: Union[Column, PendingStruct, None], dtype: Optional[DataType]) -> PendingStruct:
    if isinstance(value, PendingStruct):
        return value
    # a missing value or a non-struct one is replaced with a struct built from scratch
    return PendingStruct(value if isinstance(dtype, StructType) else None)


def _overlay_set(value, path, val, dtype):
    if not path:
        if isinstance(val, (Column, PendingStruct)):
            return val
        return val(materialize(value, dtype))

    name, _, rest = path.partition('.')
    overlay = _overlay_of(value, dtype)
    field_dtype = _field_dtype(dtype, name)
    current = None
    if name in overlay.fields or (field_dtype is not None and overlay.base is not None):
        current = overlay.get(name)
    overlay.fields[name] = _overlay_set(current, rest, val, field_dtype)
    return overlay


def _overlay_drop(value, path, dtype):
    name, _, rest = path.partition('.')
    overlay = _overlay_of(value, dtype)
    if not rest:
        overlay.fields.pop(name, None)
        return overlay

    if name in overlay.fields or overlay.base is not None:
        overlay.fields[name] = _overlay_drop(overlay.get(name), rest, _field_dtype(dtype, name))
    return overlay


def nested_get_expr(
    column: str,
    root_expr: Optional[Pending] = None,
    root_dtype: Optional[DataType] = None,
) -> Column:
    """
    Expression-level counterpart of fn.col: return the value expression of a nested column
    navigated from the given root expression, flattening array element values with fn.transform.

    :param column: column name, which can be nested (e.g. "a.b.c", "a.b[].c")
    :param root_expr: current expression for the root column (falls back to fn.col of the root)
    :param root_dtype: current data type of the root column (required when root_expr is an overlay)
    :return: value expression for the column
    """

    if isinstance(root_expr, PendingStruct) and '[]' not in column:
        value, dtype = _overlay_get(column, root_expr, root_dtype)
        return materialize(value, dtype)

    if isinstance(root_expr, PendingArrayStruct) and (path := _element_path_of(column)) is not None:
        value, dtype = _array_overlay_get(root_expr, path, _element_dtype(root_dtype))
        return materialize(value, _array_of(dtype))

    root_expr = materialize(root_expr, root_dtype)

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


def nested_is_not_null_expr(
    column: str,
    root_expr: Optional[Pending] = None,
    root_dtype: Optional[DataType] = None,
) -> Column:
    """
    Return whether a nested column is not null (an array of booleans, one per element, for array columns). A pending
    struct or array is checked through its base, so its fields are not inlined into the check.

    :param column: column name, which can be nested (e.g. "a.b.c", "a[]", "a[].b")
    :param root_expr: current expression for the root column (falls back to fn.col of the root)
    :param root_dtype: current data type of the root column (required when root_expr is an overlay)
    :return: boolean expression
    """

    if '[]' in column:
        value = None
        if isinstance(root_expr, PendingArrayStruct) and column.count('.') == 0 and column.endswith('[]'):
            value = root_expr
        elif isinstance(root_expr, PendingArrayStruct) and (path := _element_path_of(column)) is not None:
            value, _ = _array_overlay_get(root_expr, path, _element_dtype(root_dtype))
        if isinstance(value, PendingArrayStruct):
            return value.is_not_null()
        return fn.transform(nested_get_expr(column, root_expr, root_dtype), fn.isnotnull)

    if isinstance(root_expr, PendingStruct):
        value, _ = _overlay_get(column, root_expr, root_dtype)
        if isinstance(value, PendingStruct):
            return value.is_not_null()

    return fn.isnotnull(nested_get_expr(column, root_expr, root_dtype))


def _overlay_get(column, root_expr, root_dtype):
    value, dtype = root_expr, root_dtype
    for part in column.split('.')[1:]:
        value = value.getField(part) if isinstance(value, Column) else value.get(part)
        dtype = _field_dtype(dtype, part)
    return value, dtype


def _array_overlay_set(col_name, val, root_expr, root_dtype) -> Optional[PendingArrayStruct]:
    """
    Assign a field below the root array elements (e.g. "a[].b.c") in an array overlay, or return None when the column
    is not such a field.
    """

    path = _element_path_of(col_name)
    if path is None:
        return None

    parent_dtype = _element_dtype(root_dtype)
    for part in path[:-1]:
        parent_dtype = _field_dtype(parent_dtype, part)
    # a missing field can only take a plain value, as there is nothing to apply a function to
    name = path[-1]
    if not isinstance(parent_dtype, StructType) or (name not in parent_dtype.names and not isinstance(val, Column)):
        return None

    located = _array_overlay_at(root_expr, col_name.partition('[]')[0], path, root_dtype)
    if located is None:
        return None

    top, overlay, dtype = located
    if isinstance(val, Column):
        overlay.fields[name] = fn.transform(overlay.base, lambda _: val)
    else:
        current = materialize(overlay.get(name), _array_of(_field_dtype(dtype, name)))
        overlay.fields[name] = fn.transform(current, val)
    return top


def nested_set_expr(  # noqa: C901, PLR0911, PLR0915
    col_name: str,
    val: Union[Column, PendingStruct, Callable[[Column], Column]],
    root_expr: Optional[Pending],
    root_dtype: Optional[DataType],
) -> Pending:
    """
    Return the new value for the root column with the nested column set, creating a
    nested structure as needed and driven by a data type tree instead of a DataFrame schema.
    Fields of plain (non-array) structs are kept in a StructOverlay, and direct fields of the root array elements in
    an ArrayOverlay, both turned into an expression with materialize.

    :param col_name: column name, which can be nested (e.g. "a.b.c")
    :param val: value to be assigned to the column
    :param root_expr: current value of the root column (None if the column does not exist)
    :param root_dtype: current data type of the root column (None if the column does not exist)
    :return: new value for the root column
    """

    if '.' in col_name and '[]' not in col_name:
        root, _, path = col_name.partition('.')
        return _overlay_set(root_expr if root_expr is not None else fn.col(root), path, val, root_dtype)

    if (overlay := _array_overlay_set(col_name, val, root_expr, root_dtype)) is not None:
        return overlay

    root_expr = materialize(root_expr, root_dtype)
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
                fn_val = lambda _: val  # noqa: E731
            return fn.transform(base, fn_val)
        return val

    def build_struct(fields, nested, dtype, col):
        is_array = nested[0].endswith('[]')

        struct_cols = []
        for field in fields:  # noqa: PLR1702
            if field != nested[0].rstrip('[]'):
                struct_cols.append(col.getField(field).alias(field))
                continue

            if len(nested) > 1:
                field_dtype = dtype_at(dtype, nested[0])
                nested_fields = field_dtype.names

                if is_array:

                    def build_array(x):
                        return build_struct(nested_fields, nested[1:], field_dtype, x)  # noqa: B023

                    expr = fn.transform(col.getField(field), build_array)
                else:
                    expr = build_struct(nested_fields, nested[1:], field_dtype, col.getField(field))
                struct_cols.append(expr.alias(field))
            else:
                expr = val
                if is_array:
                    fn_val = val
                    if isinstance(val, Column):
                        fn_val = lambda _: val  # noqa: E731
                    expr = fn.transform(col.getField(field), fn_val).alias(field)
                elif not isinstance(val, Column):
                    expr = val(col.getField(field))

                struct_cols.append(expr.alias(field))

        if nested[0].rstrip('[]') not in fields:
            expr = val
            if is_array:
                expr = fn.array()
            elif len(nested) > 1:
                expr = build_struct([], nested[1:], None, col.getField(nested[0]))

            struct_cols.append(expr.alias(nested[0]))

        struct = fn.struct(*struct_cols)
        return struct if dtype is None else fn.when(col.isNotNull(), struct)

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


def nested_drop_expr(
    column: str,
    root_expr: Union[Column, PendingStruct],
    root_dtype: DataType,
) -> Union[Column, PendingStruct]:
    """
    Return the new value for the root column with the nested column removed.

    :param column: column name, which can be nested (e.g. "a.b", "a[].b", "a.b[].c.d")
    :param root_expr: current value of the root column
    :param root_dtype: current data type of the root column
    :return: new value for the root column
    """

    if '[]' not in column:
        return _overlay_drop(root_expr, column.partition('.')[2], root_dtype)

    path = _element_path_of(column)
    if isinstance(root_expr, PendingArrayStruct) and path is not None:
        located = _array_overlay_at(root_expr, column.partition('[]')[0], path, root_dtype)
        if located is not None:
            # the rebuilt elements only keep the fields of the data type, which no longer has the dropped one
            top, overlay, _ = located
            overlay.fields.pop(path[-1], None)
            overlay.reshaped = True
            return top

    root_expr = materialize(root_expr, root_dtype)
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
                    build_struct(field_dtype.names, nested[1:], field_dtype, col.getField(field)).alias(field)
                )

        return fn.struct(*struct_cols)

    return build_struct(root_dtype.names, nested_names, root_dtype, root_expr)


def format_value(value: Column, dtype_name: str) -> Column:
    """
    Render a value as the JSON text reported as the original next to an error.
    """

    null = fn.when(value.isNull(), fn.lit('null'))
    if dtype_name == 'string':
        return null.otherwise(fn.concat(fn.lit('"'), value, fn.lit('"')))
    if dtype_name in {'array', 'map', 'struct'}:
        return null.otherwise(fn.to_json(value, {'ignoreNullFields': 'false'}))
    return null.otherwise(value.cast('string'))
