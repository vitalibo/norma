from __future__ import annotations

import functools
from typing import Callable, Optional, Union

from pyspark.sql import Column
from pyspark.sql import functions as fn
from pyspark.sql.types import ArrayType, DataType, StructField, StructType


def dtype_at(dtype: DataType, path: str) -> DataType:
    """
    Return the data type at a path.
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
    Return a copy of the data type with the type at a path replaced, creating missing fields.
    """

    if not path:
        return new_dtype

    part, _, rest = path.partition('.')
    name = part.removesuffix('[]')
    field_dtype = _field_dtype(dtype, name)
    if part.endswith('[]'):
        field_dtype = ArrayType(dtype_set(_element_dtype(field_dtype), rest, new_dtype))
    else:
        field_dtype = dtype_set(field_dtype, rest, new_dtype)

    fields = list(dtype.fields) if isinstance(dtype, StructType) else []
    for i, field in enumerate(fields):
        if field.name == name:
            fields[i] = StructField(name, field_dtype, field.nullable, field.metadata)
            break
    else:
        fields.append(StructField(name, field_dtype, nullable=True, metadata={}))

    return StructType(fields)


def dtype_drop(dtype: DataType, path: str) -> DataType:
    """
    Return a copy of the data type without the field at a path.
    """

    part, _, rest = path.partition('.')
    name = part.removesuffix('[]')

    fields = []
    for field in dtype.fields:
        if field.name != name:
            fields.append(field)
        elif rest:
            field_dtype = field.dataType
            if part.endswith('[]'):
                field_dtype = ArrayType(dtype_drop(field_dtype.elementType, rest), field_dtype.containsNull)
            else:
                field_dtype = dtype_drop(field_dtype, rest)
            fields.append(StructField(name, field_dtype, field.nullable, field.metadata))

    return StructType(fields)


class PendingStruct:
    """
    Struct whose assigned fields are kept apart until materialized.
    """

    def __init__(self, base: Optional[Column]) -> None:
        # the current struct, or None for a struct built from scratch
        self.base = base
        self.fields = {}

    @classmethod
    def of(cls, value: Optional[Pending], dtype: Optional[DataType]) -> PendingStruct:
        """
        Wrap a value into a pending struct.
        """

        if isinstance(value, PendingStruct):
            return value
        # a missing value or a non-struct one is replaced with a struct built from scratch
        return cls(value if isinstance(dtype, StructType) else None)

    def get(self, name: str) -> Pending:
        """
        Return the current value of a field, pending if assigned deeper.
        """

        if name in self.fields:
            return self.fields[name]
        return self.base.getField(name)

    def update(
        self,
        path: str,
        new_value: Callable[[Optional[Pending], Optional[DataType]], Pending],
        dtype: Optional[DataType],
    ) -> PendingStruct:
        """
        Replace the field at a path with new_value.
        """

        name, _, rest = path.partition('.')
        field_dtype = _field_dtype(dtype, name)
        # a field missing from the data type has no current value
        current = None
        if name in self.fields or (field_dtype is not None and self.base is not None):
            current = self.get(name)
        self.fields[name] = _update_field(current, field_dtype, rest, new_value)
        return self

    def drop(self, path: str, dtype: Optional[DataType]) -> PendingStruct:
        """
        Drop the field at a path.
        """

        name, _, rest = path.partition('.')
        if rest:
            field_dtype = _field_dtype(dtype, name)
            self.fields[name] = PendingStruct.of(self.get(name), field_dtype).drop(rest, field_dtype)
        else:
            self.fields.pop(name, None)
        return self

    def materialize(self, dtype: Optional[DataType]) -> Column:
        """
        Build the struct with fields in the data type order.
        """

        names = list(dtype.names) if isinstance(dtype, StructType) else []
        names += [name for name in self.fields if name not in names]
        struct = fn.struct(*(materialize(self.get(name), _field_dtype(dtype, name)).alias(name) for name in names))
        return struct if self.base is None else fn.when(self.base.isNotNull(), struct)

    def is_not_null(self) -> Column:
        """
        Return whether the struct is not null, without building it.
        """

        return fn.lit(True) if self.base is None else self.base.isNotNull()


class PendingArrayStruct:
    """
    Array of structs whose assigned element fields are kept apart, as arrays aligned with the base array.
    """

    def __init__(self, base: Column) -> None:
        self.base = base
        self.fields = {}
        # set once a field is dropped: the elements must then be rebuilt even without any assigned field
        self.reshaped = False

    @classmethod
    def of(cls, value: Pending, dtype: Optional[DataType]) -> PendingArrayStruct:
        """
        Wrap a value into a pending array.
        """

        return value if isinstance(value, PendingArrayStruct) else cls(materialize(value, dtype))

    def get(self, name: str) -> Union[Column, PendingArrayStruct]:
        """
        Return the values of an element field, as an array aligned with the base.
        """

        if name in self.fields:
            return self.fields[name]
        return fn.transform(self.base, lambda x: x.getField(name))

    def assign(
        self, path: str, val: Union[Column, Callable[[Column], Column]], element_dtype: Optional[DataType],
    ) -> PendingArrayStruct:
        """
        Assign the element field at a path, with a value or a function of the current one.
        """

        parent, _, name = path.rpartition('.')
        pending = self._nested(parent)
        if isinstance(val, Column):
            pending.fields[name] = fn.transform(pending.base, lambda _: val)
        else:
            current = materialize(pending.get(name), _array_of(_field_dtype(element_dtype, path)))
            pending.fields[name] = fn.transform(current, val)
        return self

    def drop(self, path: str) -> PendingArrayStruct:
        """
        Drop the element field at a path.
        """

        parent, _, name = path.rpartition('.')
        pending = self._nested(parent)
        pending.fields.pop(name, None)
        # the rebuilt elements only keep the fields of the data type, which no longer has the dropped one
        pending.reshaped = True
        return self

    def materialize(self, dtype: Optional[DataType]) -> Column:
        """
        Build the array with element fields in the data type order.
        """

        if not self.fields and not self.reshaped:
            return self.base

        element_dtype = _element_dtype(dtype)
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
        Return whether each element is not null, without building the elements.
        """

        return fn.transform(self.base, fn.isnotnull)

    def _nested(self, path: str) -> PendingArrayStruct:
        """
        Return the pending array of a nested element struct, making every struct along the path pending.
        """

        pending = self
        for name in path.split('.') if path else []:
            nested = pending.fields.get(name)
            if not isinstance(nested, PendingArrayStruct):
                nested = pending.fields[name] = PendingArrayStruct(pending.get(name))
            pending = nested
        return pending


Pending = Union[Column, PendingStruct, PendingArrayStruct]


def materialize(value: Optional[Pending], dtype: Optional[DataType]) -> Optional[Column]:
    """
    Turn a pending value into an expression.
    """

    return value.materialize(dtype) if isinstance(value, (PendingStruct, PendingArrayStruct)) else value


def nested_get_expr(
    column: str,
    root_expr: Optional[Pending] = None,
    root_dtype: Optional[DataType] = None,
) -> Column:
    """
    Return the value of a nested column, an array of values below array elements.
    """

    return materialize(*_resolve(column, root_expr, root_dtype))


def nested_is_not_null_expr(
    column: str,
    root_expr: Optional[Pending] = None,
    root_dtype: Optional[DataType] = None,
) -> Column:
    """
    Return whether a nested column is not null (per element below array elements).
    """

    value, dtype = _resolve(column, root_expr, root_dtype)
    if '[]' in column:
        if isinstance(value, PendingArrayStruct):
            return value.is_not_null()
        return fn.transform(materialize(value, dtype), fn.isnotnull)

    if isinstance(value, PendingStruct):
        return value.is_not_null()
    return fn.isnotnull(materialize(value, dtype))


def nested_set_expr(
    column: str,
    val: Union[Column, PendingStruct, Callable[[Column], Column]],
    root_expr: Optional[Pending],
    root_dtype: Optional[DataType],
) -> Pending:
    """
    Return the new value of the root column with a nested column set, creating missing structs.
    """

    root, path, elements = _split(column)

    def new_value(current, dtype):
        if elements is None:
            return val if isinstance(val, (Column, PendingStruct)) else val(materialize(current, dtype))

        if dtype is None:
            # a missing array is created empty
            return fn.array()

        if not elements:
            return fn.transform(materialize(current, dtype), (lambda _: val) if isinstance(val, Column) else val)

        element_dtype = _element_dtype(dtype)
        parent, _, name = elements.rpartition('.')
        parent_dtype = _field_dtype(element_dtype, parent)
        # a missing field is only kept pending with a plain value, as there is nothing to apply a function to
        if isinstance(parent_dtype, StructType) and (isinstance(val, Column) or name in parent_dtype.names):
            return PendingArrayStruct.of(current, dtype).assign(elements, val, element_dtype)

        # otherwise, the field is set in every element rebuilt on its own
        return fn.transform(
            materialize(current, dtype),
            lambda x: materialize(nested_set_expr(f'_.{elements}', val, x, element_dtype), element_dtype),
        )

    return _update_field(fn.col(root) if root_expr is None else root_expr, root_dtype, path, new_value)


def nested_drop_expr(
    column: str,
    root_expr: Pending,
    root_dtype: DataType,
) -> Pending:
    """
    Return the new value of the root column with a nested column removed.
    """

    _, path, elements = _split(column)
    if elements is None:
        return PendingStruct.of(root_expr, root_dtype).drop(path, root_dtype)

    def drop(x, x_dtype, names):
        name, *rest = names
        if rest:
            return x.withField(name, drop(x[name], _field_dtype(x_dtype, name), rest))
        if isinstance(x_dtype, StructType) and x_dtype.names == [name]:
            # dropFields can't remove the last field of a struct
            return fn.when(x.isNotNull(), fn.struct())
        return x.dropFields(name)

    def drop_elements(current, dtype):
        if isinstance(current, PendingArrayStruct):
            return current.drop(elements)
        return fn.transform(materialize(current, dtype), lambda x: drop(x, _element_dtype(dtype), elements.split('.')))

    return _update_field(root_expr, root_dtype, path, drop_elements)


def elementwise(column: str, func: Callable[..., Column], *cols: Column) -> Column:
    """
    Apply a function to the values of a column, element by element below array elements.
    """

    if '[]' not in column:
        return func(*cols)
    return fn.transform(*cols, func) if len(cols) == 1 else fn.zip_with(*cols, func)


def format_value(value: Column, dtype_name: str) -> Column:
    """
    Render a value as JSON text.
    """

    null = fn.when(value.isNull(), fn.lit('null'))
    if dtype_name == 'string':
        return null.otherwise(fn.concat(fn.lit('"'), value, fn.lit('"')))
    if dtype_name in {'array', 'map', 'struct'}:
        return null.otherwise(fn.to_json(value, {'ignoreNullFields': 'false'}))
    return null.otherwise(value.cast('string'))


def _split(column: str) -> tuple[str, str, Optional[str]]:
    """
    Split a column into its root, the path below it and the path below array elements.
    """

    head, sep, rest = column.partition('[]')
    root, _, path = head.partition('.')
    return root, path, rest.removeprefix('.') if sep else None


def _resolve(
    column: str,
    root_expr: Optional[Pending],
    root_dtype: Optional[DataType],
) -> tuple[Pending, Optional[DataType]]:
    """
    Return the pending value of a column and its data type (an array of values below array elements).
    """

    root, path, elements = _split(column)
    value = fn.col(root) if root_expr is None else root_expr
    for name in path.split('.') if path else []:
        value = value.getField(name) if isinstance(value, Column) else value.get(name)
    dtype = _field_dtype(root_dtype, path)
    if not elements:
        return value, dtype

    dtype = _array_of(_field_dtype(_element_dtype(dtype), elements))
    while elements and isinstance(value, PendingArrayStruct):
        name, _, elements = elements.partition('.')
        value = value.get(name)
    if elements:
        names = elements.split('.')
        value = fn.transform(value, lambda x: functools.reduce(Column.getField, names, x))
    return value, dtype


def _update_field(
    value: Optional[Pending],
    dtype: Optional[DataType],
    path: str,
    new_value: Callable[[Optional[Pending], Optional[DataType]], Pending],
) -> Pending:
    """
    Replace the value at a path with new_value(current value, current data type).
    """

    if not path:
        return new_value(value, dtype)
    return PendingStruct.of(value, dtype).update(path, new_value, dtype)


def _field_dtype(dtype: Optional[DataType], path: str) -> Optional[DataType]:
    """
    Return the data type at a struct path, or None when missing.
    """

    for name in path.split('.') if path else []:
        dtype = dtype[name].dataType if isinstance(dtype, StructType) and name in dtype.names else None
    return dtype


def _element_dtype(dtype: Optional[DataType]) -> Optional[DataType]:
    return dtype.elementType if isinstance(dtype, ArrayType) else None


def _array_of(dtype: Optional[DataType]) -> Optional[DataType]:
    return None if dtype is None else ArrayType(dtype)
