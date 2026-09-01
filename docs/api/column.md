# Column

The `norma.schema.Column` class describes a single field in a `norma.schema.Schema`.
It bundles together the expected data type, validation rules, nullability and default handling for a column.
Columns are typically instantiated inline when constructing a `norma.schema.Schema`.

## Constructor

```python
Column(
    dtype,
    *,
    rules=None,
    required=False,
    nullable=False,
    eq=None,
    ne=None,
    gt=None,
    lt=None,
    ge=None,
    le=None,
    multiple_of=None,
    min_length=None,
    max_length=None,
    pattern=None,
    isin=None,
    notin=None,
    min_items=None,
    max_items=None,
    unique_items=False,
    inner_schema=None,
    default=UNSET,
)
```

#### Parameters

- `dtype`: Python type or string alias describing the expected data type. Aliases mirror the built-in rules (e.g. `int`,
  `"integer"`, `"array"`). Nested types such as `"object"` or `"array"` require `inner_schema`.
- `rules`: Optional `Rule` instance or list of rules for extra validation logic. Custom rules run after built-in
  constraints and must have unique names.
- `required`: Whether the field must be carried by the input. A field the input never carried is reported as
  `missing`, unless the column has a `default`, which answers the absence instead.
- `nullable`: Whether a null value is accepted. This governs the value, not its presence: a field that is absent is
  never a type error, whatever `nullable` says.
- `eq`, `ne`: Scalar equality or inequality constraints.
- `gt`, `lt`, `ge`, `le`: Scalar comparison constraints (greater than, less than, greater than or equal to, less than or
  equal to).
- `multiple_of`: Numeric constraint ensuring the value is a multiple of the given number.
- `min_length`, `max_length`: String length constraints.
- `pattern`: Regular expression pattern that string values must match.
- `isin`, `notin`: Membership constraints based on iterable inputs.
- `min_items`, `max_items`, `unique_items`: Sequence-specific constraints applied to arrays or lists.
- `inner_schema`: Nested `norma.schema.Schema` (for objects) or `norma.schema.Column` (for array item definitions).
- `default`: Value injected where the field is **absent** from the input.

### Examples

```python
from norma.schema import Column, Schema

schema = Schema({
    'id': Column(int, required=True),
    'email': Column(str, pattern=r"^\\S+@\\S+$"),
    'nickname': Column(str, nullable=True),
    'status': Column(str, isin=['active', 'banned'], default='active'),
    'addresses': Column('array', inner_schema=Column('object', inner_schema=Schema({
        'street': Column(str),
        'city': Column(str, required=True)
    })))
})
```

The constructor validates that the requested `dtype` is supported and raises `ValueError` for unsupported types, for a
nested type given without an `inner_schema`, and for a `default` that cannot mean anything - one on array items, or a
non-null one on a container column.
