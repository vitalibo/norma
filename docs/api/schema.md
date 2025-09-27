# Schema

The `norma.schema.Schema` class coordinates column definitions and orchestrates validation across entire tabular
datasets. A schema is a mapping of column names to `norma.schema.Column` instances combined with configuration
controlling how extra fields are handled.

## Constructor

```python
Schema(
    columns,
    allow_extra=False,
)
```

#### Parameters

- `columns`: `dict[str, Column]` - Dictionary mapping column names to `Column` instances that define type,
  constraints and defaults for each field.
- `allow_extra`: `bool` - When `False`, unexpected columns in the input data will trigger validation errors.
  When `True`, unexpected columns are ignored.

## Methods

### validate

Validates a Pandas/PySpark `DataFrame` against the schema and returns the input `DataFrame` with an additional column
containing validation results.

```python
validate(
    df,
    error_col='errors',
)
```

#### Parameters

- `df`: `DataFrame` - Input DataFrame. Only Pandas and PySpark engines are supported.
- `error_col`: `str` - Optional column name where validation errors are collected. Defaults to `"errors"`.

#### Returns

- `DataFrame` - Input DataFrame with an additional column containing validation results. When `allow_extra=False`,
  only schema-defined columns (plus the error column) are returned; unexpected inputs are dropped.

#### Raises

- `NotImplementedError` - Raises when the provided engine is unsupported.

### from_json_schema

Class method that constructs a `Schema` from a JSON Schema dictionary.

```python
from_json_schema(
    json_schema
)
```

#### Parameters

- `json_schema`: `dict` - JSON Schema dictionary defining the structure and constraints of the data.

#### Returns

- `Schema` - Constructed `norma.schema.Schema` instance.
