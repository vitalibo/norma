# Rules reference

This section provides a comprehensive reference for all rules available in Norma.
Each rule is documented with its purpose, options, and examples of correct and incorrect code.

## Data Type Conventions

In JSON Schema, the data type of property is defined using the `type` keyword.
Norma's rules enforce consistent type conventions across your datasets.
Unlike JSON Schema, Norma attempts to coerce incoming values into the specified type before validation.

- `array`
- `boolean`
- `integer`
- `number`
- `object`
- `string`

Each section below includes a conversion table that shows sample inputs, their inferred types, the resulting values,
and any errors raised during coercion.

These types have corresponding engine-specific types:

| JSON    | Python API | Pandas         | PySpark |
|---------|------------|----------------|---------|
| array   | list       | object         | array   |
| boolean | bool       | boolean        | boolean |
| integer | int        | Int64          | long    |
| number  | float      | Float64        | float   |
| object  | object     | object         | struct  |
| string  | str        | string[python] | string  |

Unlike JSON Schema, Norma does not fully support union types (i.e. `type` as an array).
The exception is `null`, which makes a property nullable, as described in [Null values](#null-values).

Here are some examples of data type rules defined in JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "age": {
      "type": "integer"
    },
    "nickname": {
      "type": [
        "string",
        "null"
      ]
    }
  }
}
```

Otherwise, the rules can be defined using the Python API:

```python
from norma.schema import Schema, Column

schema = Schema({
    'name': Column(str),
    'age': Column(int),
    'nickname': Column(str, nullable=True)
})
```

#### Null values

A `null` passes only a nullable property. Nullability is part of the type: add `null` to `type` in JSON Schema, or pass
`nullable=True` in the Python API. A property is not nullable unless it says so.

- **Not nullable**: a `null` is reported with the type error of the declared type (`bool_type`, `int_type`,
  `string_type`, …) and stays `null` in the output.
- **Nullable**: a `null` passes without an error.

| Input  | Type             | Output | Error         | Reason                         |
|--------|------------------|--------|---------------|--------------------------------|
| `null` | `string`         | `null` | `string_type` | Input should be a valid string |
| `null` | `[string, null]` | `null` |               |                                |
| `foo`  | `string`         | `foo`  |               |                                |
| `foo`  | `[string, null]` | `foo`  |               |                                |

#### `boolean`

The `boolean` type is used to represent true/false values.

**JSON Schema**

```json
{
  "type": "boolean"
}
```

**Python**

```python
from norma.schema import Column

Column(bool)
```

Norma accepts common truthy and falsy strings or numbers when coercing to `boolean`.

Use the table below to coerce other types to `boolean`:

| Input                                                                  | Type                | Output  | Error          | Reason                                                     |
|------------------------------------------------------------------------|---------------------|---------|----------------|------------------------------------------------------------|
| `null`                                                                 | `boolean`           | `null`  | `bool_type`    | Input should be a valid boolean                            |
| `null`                                                                 | `[boolean, null]`   | `null`  |                |                                                            |
| `true`                                                                 | `boolean`           | `true`  |                |                                                            |
| `false`                                                                | `boolean`           | `false` |                |                                                            |
| `"true"`<br>`"True"`<br>`"t"`<br>`"on"`<br>`"yes"`<br>`"y"`<br>`"1"`   | `string`            | `true`  |                |                                                            |
| `"false"`<br>`"False"`<br>`"f"`<br>`"off"`<br>`"no"`<br>`"n"`<br>`"0"` | `string`            | `false` |                |                                                            |
| `1`<br>`123`                                                           | `integer`           | `true`  |                |                                                            |
| `0`                                                                    | `integer`           | `false` |                |                                                            |
| `1.23`<br>`0.123`                                                      | `number`            | `true`  |                |                                                            |
| `0.0`                                                                  | `number`            | `false` |                |                                                            |
| `"foo"`                                                                | `string`            | `null`  | `bool_parsing` | Input should be a valid boolean, unable to interpret input |
| `{"foo":"bar"}`<br>`["foo","bar"]`                                     | `object`<br>`array` | `null`  | `bool_type`    | Input should be a valid boolean                            |

#### `integer`

The `integer` type is used to represent whole numbers.

**JSON Schema**

```json
{
  "type": "integer"
}
```

**Python**

```python
from norma.schema import Column

Column(int)
```

Use the table below to coerce other types to `integer`:

| Input                              | Type                | Output | Error         | Reason                                                                |
|------------------------------------|---------------------|--------|---------------|-----------------------------------------------------------------------|
| `null`                             | `integer`           | `null` | `int_type`    | Input should be a valid integer                                       |
| `null`                             | `[integer, null]`   | `null` |               |                                                                       |
| `123`                              | `integer`           | `123`  |               |                                                                       |
| `123.0`                            | `number`            | `123`  |               |                                                                       |
| `123.45`                           | `number`            | `123`  |               |                                                                       |
| `"123"`                            | `string`            | `123`  |               |                                                                       |
| `"123.45"`                         | `string`            | `123`  |               |                                                                       |
| `true`                             | `boolean`           | `1`    |               |                                                                       |
| `false`                            | `boolean`           | `0`    |               |                                                                       |
| `"foo"`                            | `string`            | `null` | `int_parsing` | Input should be a valid integer, unable to parse string as an integer |
| `{"foo":"bar"}`<br>`["foo","bar"]` | `object`<br>`array` | `null` | `int_type`    | Input should be a valid integer                                       |

#### `number`

The `number` type is used to represent floating-point numbers.

**JSON Schema**

```json
{
  "type": "number"
}
```

**Python**

```python
from norma.schema import Column

Column(float)
```

Use the table below to coerce other types to `number`:

| Input                              | Type                | Output   | Error           | Reason                                                             |
|------------------------------------|---------------------|----------|-----------------|--------------------------------------------------------------------|
| `null`                             | `number`            | `null`   | `float_type`    | Input should be a valid number                                     |
| `null`                             | `[number, null]`    | `null`   |                 |                                                                    |
| `123.45`                           | `number`            | `123.45` |                 |                                                                    |
| `123`                              | `integer`           | `123.0`  |                 |                                                                    |
| `"123.45"`                         | `string`            | `123.45` |                 |                                                                    |
| `"123"`                            | `string`            | `123.0`  |                 |                                                                    |
| `true`                             | `boolean`           | `1.0`    |                 |                                                                    |
| `false`                            | `boolean`           | `0.0`    |                 |                                                                    |
| `"foo"`                            | `string`            | `null`   | `float_parsing` | Input should be a valid number, unable to parse string as a number |
| `{"foo":"bar"}`<br>`["foo","bar"]` | `object`<br>`array` | `null`   | `float_type`    | Input should be a valid number                                     |

#### `string`

The `string` type is used to represent text data.

**JSON Schema**

```json
{
  "type": "string"
}
```

**Python**

```python
from norma.schema import Column

Column(str)
```

Non-string inputs are stringified using their canonical representations when coercion succeeds.

Use the table below to coerce other types to `string`:

| Input                              | Type                | Output                   | Error         | Reason                         |
|------------------------------------|---------------------|--------------------------|---------------|--------------------------------|
| `null`                             | `string`            | `null`                   | `string_type` | Input should be a valid string |
| `null`                             | `[string, null]`    | `null`                   |               |                                |
| `"foo"`                            | `string`            | `"foo"`                  |               |                                |
| `123`                              | `integer`           | `"123"`                  |               |                                |
| `123.45`                           | `number`            | `"123.45"`               |               |                                |
| `true`                             | `boolean`           | `"true"`                 |               |                                |
| `2023-10-05T14:48:00Z`             | `timestamp`         | `"2023-10-05T14:48:00Z"` |               |                                |
| `{"foo":"bar"}`<br>`["foo","bar"]` | `object`<br>`array` | `null`                   | `string_type` | Input should be a valid string |

#### `array`

The `array` type is used to represent ordered lists of values of the same type.

Array elements can be any supported Norma type except another `array`; nested arrays are not yet supported.

**JSON Schema**

```json
{
  "type": "array",
  "items": {
    "type": "string"
  }
}
```

**Python**

```python
from norma.schema import Column

Column(list, inner_schema=Column(str))
```

JSON-formatted strings are parsed into arrays when possible; other scalar types raise errors.

Use the table below to coerce other types to `array`:

| Input                                         | Type                                           | Output          | Error           | Reason                                                            |
|-----------------------------------------------|------------------------------------------------|-----------------|-----------------|-------------------------------------------------------------------|
| `null`                                        | `array`                                        | `null`          | `array_type`    | Input should be a valid array                                     |
| `null`                                        | `[array, null]`                                | `null`          |                 |                                                                   |
| `["foo","bar"]`                               | `array`                                        | `["foo","bar"]` |                 |                                                                   |
| `'["foo","bar"]'`                             | `string`                                       | `["foo","bar"]` |                 |                                                                   |
| `"[1, 2]"`                                    | `string`                                       | `["1","2"]`     |                 |                                                                   |
| `"foo"`                                       | `string`                                       | `null`          | `array_parsing` | Input should be a valid array, unable to parse string as an array |
| `123`<br>`1.23`<br>`false`<br>`{"foo":"bar"}` | `integer`<br>`number`<br>`boolean`<br>`object` | `null`          | `array_type`    | Input should be a valid array                                     |

#### `object`

The `object` type is used to represent structured data with key-value pairs.

**JSON Schema**

```json
{
  "type": "object",
  "properties": {
    "key": {
      "type": "string"
    }
  }
}
```

**Python**

```python
from norma.schema import Schema, Column

Column(object, inner_schema=Schema({
    'key': Column(str)
}))
```

JSON-formatted strings are parsed into objects when possible; non-mapping inputs raise type errors.

Use the table below to coerce other types to `object`:

| Input                                         | Type                                          | Output          | Error            | Reason                                                              |
|-----------------------------------------------|-----------------------------------------------|-----------------|------------------|---------------------------------------------------------------------|
| `null`                                        | `object`                                      | `null`          | `object_type`    | Input should be a valid object                                      |
| `null`                                        | `[object, null]`                              | `null`          |                  |                                                                     |
| `{"foo":"bar"}`                               | `struct`<br>`map`                             | `{"foo":"bar"}` |                  |                                                                     |
| `'{"foo":"bar"}'`                             | `string`                                      | `{"foo":"bar"}` |                  |                                                                     |
| `"foo"`                                       | `string`                                      | `null`          | `object_parsing` | Input should be a valid object, unable to parse string as an object |
| `123`<br>`1.23`<br>`false`<br>`["foo","bar"]` | `integer`<br>`number`<br>`boolean`<br>`array` | `null`          | `object_type`    | Input should be a valid object                                      |

## Data Types with Formats

In addition to the basic data types, Norma supports the `format` keyword to capture semantic information about a value.
Formats extend the base `string` type to validate specific patterns or structures (for example, RFC 3339 timestamps).

All format types are string-based and add stricter validation to the main type.
Specify the format with the `format` keyword in JSON Schema or by using the corresponding type name in the Python API.

Norma supports the following format types based on JSON Schema:

- `date-time`
- `date`
- `time`
- `duration`
- `ipv4`
- `ipv6`
- `uri`
- `uuid`

These types have corresponding engine-specific types:

| JSON      | Python API | Pandas              | PySpark   |
|-----------|------------|---------------------|-----------|
| date-time | datetime   | datetime64[ns, UTC] | timestamp |
| date      | date       | datetime64[s]       | date      |
| time      | time       | string[python]      | string    |
| duration  | duration   | string[python]      | string    |
| ipv4      | str        | string[python]      | string    |
| ipv6      | str        | string[python]      | string    |
| uri       | str        | string[python]      | string    |
| uuid      | str        | string[python]      | string    |

Here are some examples of format rules defined in JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "format": "uuid"
    },
    "ip": {
      "type": "string",
      "format": "ipv4"
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

### Dates, Times, and Duration

#### `date-time`

The `date-time` format represents date and time information ("date-time" production) according to RFC 3339.
It's used for timestamps with full date and time precision.

**JSON Schema**

```json
{
  "type": "string",
  "format": "date-time"
}
```

**Python**

```python
from norma.schema import Column

Column('datetime')
```

Use the table below to coerce other types to `date-time`:

| Input                                                 | Type                                          | Output                       | Error              | Reason                                                                 |
|-------------------------------------------------------|-----------------------------------------------|------------------------------|--------------------|------------------------------------------------------------------------|
| `null`                                                | `string`                                      | `null`                       | `datetime_type`    | Input should be a valid datetime                                       |
| `null`                                                | `[string, null]`                              | `null`                       |                    |                                                                        |
| `"2025-01-23T12:34:56Z"`                              | `string`                                      | `"2025-01-23T12:34:56.000Z"` |                    |                                                                        |
| `"2025-01-23 12:34:56"`                               | `string`                                      | `"2025-01-23T12:34:56.000Z"` |                    |                                                                        |
| `"2025-01-23"`                                        | `date`                                        | `"2025-01-23T00:00:00.000Z"` |                    |                                                                        |
| `"foo"`                                               | `string`                                      | `null`                       | `datetime_parsing` | Input should be a valid datetime, unable to parse string as a datetime |
| `123`<br>`true`<br>`{"foo":"bar"}`<br>`["foo","bar"]` | `integer`<br>`boolean`<br>`object`<br>`array` | `null`                       | `datetime_type`    | Input should be a valid datetime                                       |

#### `date`

The `date` format represents "full-date" production according to RFC 3339.

**JSON Schema**

```json
{
  "type": "string",
  "format": "date"
}
```

**Python**

```python
from norma.schema import Column

Column('date')
```

Use the table below to coerce other types to `date`:

| Input                                                 | Type                                          | Output         | Error          | Reason                                                         |
|-------------------------------------------------------|-----------------------------------------------|----------------|----------------|----------------------------------------------------------------|
| `null`                                                | `string`                                      | `null`         | `date_type`    | Input should be a valid date                                   |
| `null`                                                | `[string, null]`                              | `null`         |                |                                                                |
| `"2025-10-05"`                                        | `string`                                      | `"2025-10-05"` |                |                                                                |
| `"2025-10-05"`                                        | `date`                                        | `"2025-10-05"` |                |                                                                |
| `"2025-10-05T14:48:00Z"`                              | `string`                                      | `"2025-10-05"` |                |                                                                |
| `"foo"`                                               | `string`                                      | `null`         | `date_parsing` | Input should be a valid date, unable to parse string as a date |
| `123`<br>`true`<br>`{"foo":"bar"}`<br>`["foo","bar"]` | `integer`<br>`boolean`<br>`object`<br>`array` | `null`         | `date_type`    | Input should be a valid date                                   |

#### `time`

The `time` format represents "full-time" production according to RFC 3339.

**JSON Schema**

```json
{
  "type": "string",
  "format": "time"
}
```

**Python**

```python
from norma.schema import Column

Column('time')
```

Use the table below to coerce other types to `time`:

| Input                                                 | Type                                          | Output            | Error          | Reason                                                                   |
|-------------------------------------------------------|-----------------------------------------------|-------------------|----------------|--------------------------------------------------------------------------|
| `null`                                                | `string`                                      | `null`            | `time_type`    | Input should be a valid time                                             |
| `null`                                                | `[string, null]`                              | `null`            |                |                                                                          |
| `"14:48:00Z"`                                         | `string`                                      | `"14:48:00Z"`     |                |                                                                          |
| `"14:48:00"`                                          | `string`                                      | `"14:48:00"`      |                |                                                                          |
| `"14:48:00.123Z"`                                     | `string`                                      | `"14:48:00.123Z"` |                |                                                                          |
| `"foo"`                                               | `string`                                      | `null`            | `time_parsing` | Input should be in a valid time format, unable to parse string as a time |
| `123`<br>`true`<br>`{"foo":"bar"}`<br>`["foo","bar"]` | `integer`<br>`boolean`<br>`object`<br>`array` | `null`            | `time_type`    | Input should be a valid time                                             |

#### `duration`

The `duration` format represents "duration" production according to RFC 3339.

**JSON Schema**

```json
{
  "type": "string",
  "format": "duration"
}
```

**Python**

```python
from norma.schema import Column

Column('duration')
```

Use the table below to coerce other types to `duration`:

| Input                                                 | Type                                          | Output    | Error              | Reason                                                                 |
|-------------------------------------------------------|-----------------------------------------------|-----------|--------------------|------------------------------------------------------------------------|
| `null`                                                | `string`                                      | `null`    | `duration_type`    | Input should be a valid duration                                       |
| `null`                                                | `[string, null]`                              | `null`    |                    |                                                                        |
| `"PT30M"`                                             | `string`                                      | `"PT30M"` |                    |                                                                        |
| `"P1D"`                                               | `string`                                      | `"P1D"`   |                    |                                                                        |
| `"foo"`                                               | `string`                                      | `null`    | `duration_parsing` | Input should be a valid duration, unable to parse string as a duration |
| `123`<br>`true`<br>`{"foo":"bar"}`<br>`["foo","bar"]` | `integer`<br>`boolean`<br>`object`<br>`array` | `null`    | `duration_type`    | Input should be a valid duration                                       |

### IP Addresses

#### `ipv4`

The `ipv4` address according to the "dotted-quad" ABNF syntax as defined in RFC 2673.

**JSON Schema**

```json
{
  "type": "string",
  "format": "ipv4"
}
```

**Python**

```python
from norma.schema import Column

Column('ipv4')
```

Use the table below to coerce other types to `ipv4`:

| Input                                                 | Type                                          | Output          | Error  | Reason                            |
|-------------------------------------------------------|-----------------------------------------------|-----------------|--------|-----------------------------------|
| `null`                                                | `string`                                      | `null`          | `ipv4` | Input is not a valid IPv4 address |
| `null`                                                | `[string, null]`                              | `null`          |        |                                   |
| `"192.168.1.1"`                                       | `string`                                      | `"192.168.1.1"` |        |                                   |
| `"foo"`                                               | `string`                                      | `null`          | `ipv4` | Input is not a valid IPv4 address |
| `123`<br>`true`<br>`{"foo":"bar"}`<br>`["foo","bar"]` | `integer`<br>`boolean`<br>`object`<br>`array` | `null`          | `ipv4` | Input is not a valid IPv4 address |

#### `ipv6`

The `ipv6` format validates compressed or full IPv6 literals as defined in RFC 4291.

**JSON Schema**

```json
{
  "type": "string",
  "format": "ipv6"
}
```

**Python**

```python
from norma.schema import Column

Column('ipv6')
```

Use the table below to coerce other types to `ipv6`:

| Input                                                 | Type                                          | Output                             | Error  | Reason                            |
|-------------------------------------------------------|-----------------------------------------------|------------------------------------|--------|-----------------------------------|
| `null`                                                | `string`                                      | `null`                             | `ipv6` | Input is not a valid IPv6 address |
| `null`                                                | `[string, null]`                              | `null`                             |        |                                   |
| `"2001:0db8:85a3::8a2e:0370:7334"`                    | `string`                                      | `"2001:0db8:85a3::8a2e:0370:7334"` |        |                                   |
| `"foo"`                                               | `string`                                      | `null`                             | `ipv6` | Input is not a valid IPv6 address |
| `123`<br>`true`<br>`{"foo":"bar"}`<br>`["foo","bar"]` | `integer`<br>`boolean`<br>`object`<br>`array` | `null`                             | `ipv6` | Input is not a valid IPv6 address |

### Resource Identifiers

#### `uri`

The `uri` format validates Uniform Resource Identifiers as defined in RFC 3986 (absolute or relative with optional query
and fragment).

**JSON Schema**

```json
{
  "type": "string",
  "format": "uri"
}
```

**Python**

```python
from norma.schema import Column

Column('uri')
```

Use the table below to coerce other types to `uri`:

| Input                                                 | Type                                          | Output                           | Error         | Reason                                                       |
|-------------------------------------------------------|-----------------------------------------------|----------------------------------|---------------|--------------------------------------------------------------|
| `null`                                                | `string`                                      | `null`                           | `uri_type`    | Input should be a valid URI                                  |
| `null`                                                | `[string, null]`                              | `null`                           |               |                                                              |
| `"https://example.com/path?q=v"`                      | `string`                                      | `"https://example.com/path?q=v"` |               |                                                              |
| `"ftp://files.example.com"`                           | `string`                                      | `"ftp://files.example.com"`      |               |                                                              |
| `"foo"`                                               | `string`                                      | `null`                           | `uri_parsing` | Input should be a valid URI, unable to parse string as a URI |
| `123`<br>`true`<br>`{"foo":"bar"}`<br>`["foo","bar"]` | `integer`<br>`boolean`<br>`object`<br>`array` | `null`                           | `uri_type`    | Input should be a valid URI                                  |

#### `uuid`

The `uuid` format represents UUIDv4 (Universally Unique Identifier) according to RFC 4122.

**JSON Schema**

```json
{
  "type": "string",
  "format": "uuid"
}
```

**Python**

```python
from norma.schema import Column

Column('uuid')
```

Use the table below to coerce other types to `uuid`:

| Input                                                 | Type                                          | Output                                   | Error          | Reason                                                         |
|-------------------------------------------------------|-----------------------------------------------|------------------------------------------|----------------|----------------------------------------------------------------|
| `null`                                                | `string`                                      | `null`                                   | `uuid_type`    | UUID input should be a string                                  |
| `null`                                                | `[string, null]`                              | `null`                                   |                |                                                                |
| `"b9e95281-2f6b-4f75-8010-5afabaca81e7"`              | `string`                                      | `"b9e95281-2f6b-4f75-8010-5afabaca81e7"` |                |                                                                |
| `"foo"`                                               | `string`                                      | `null`                                   | `uuid_parsing` | Input should be a valid UUID, unable to parse string as a UUID |
| `123`<br>`true`<br>`{"foo":"bar"}`<br>`["foo","bar"]` | `integer`<br>`boolean`<br>`object`<br>`array` | `null`                                   | `uuid_type`    | UUID input should be a string                                  |

## Value constraints

In addition to data type conversions, Norma supports various constraints to enforce specific rules on the values of
properties.

These constraints can be applied to different data types and include:

- `required`
- `nullable`
- `default`
- `additionalProperties`
- `enum`
- `const`
- `minimum` / `exclusiveMinimum`
- `maximum` / `exclusiveMaximum`
- `minLength` / `maxLength`
- `pattern`
- `multipleOf`
- `minItems` / `maxItems`
- `uniqueItems`
- `not`

The examples that follow demonstrate each constraint in JSON Schema and the Python API, alongside representative inputs
so you can see how Norma evaluates real data.

A constraint never checks a `null`, so the `null` rows in these tables assume a nullable property. See
[Null values](#null-values) for a property that is not nullable.

### Presence and absence

`required`, `nullable` and `default` answer three different questions, and Norma keeps them apart. A field reaches the
validator in one of three states:

| Input  | State     | What it means                                      |
|--------|-----------|----------------------------------------------------|
| `17`   | `present` | The source carried the field, with a value         |
| `null` | `null`    | The source carried the field, and said it is empty |
|        | `absent`  | The source never carried the field                 |

Every combination of the three states and the three keywords, plus a value that fails its own rules for a property
declared `{"type": "integer"}`.

| Input   | `required`        | `nullable`        | `default`           | Output | Error         | Reason                                                                |
|---------|-------------------|-------------------|---------------------|--------|---------------|-----------------------------------------------------------------------|
| `17`    | `false`<br>`true` | `false`<br>`true` | -<br>`25`<br>`null` | `17`   |               |                                                                       |
| `"abc"` | `false`<br>`true` | `false`<br>`true` | -<br>`25`<br>`null` | `null` | `int_parsing` | Input should be a valid integer, unable to parse string as an integer |
| `null`  | `false`<br>`true` | `false`           | -<br>`25`<br>`null` | `null` | `int_type`    | Input should be a valid integer                                       |
| `null`  | `false`<br>`true` | `true`            | -<br>`25`<br>`null` | `null` |               |                                                                       |
|         | `false`           | `false`<br>`true` | -                   |        |               |                                                                       |
|         | `false`<br>`true` | `false`<br>`true` | `25`                | `25`   |               |                                                                       |
|         | `false`<br>`true` | `false`           | `null`              | `null` | `int_type`    | Input should be a valid integer                                       |
|         | `false`<br>`true` | `true`            | `null`              | `null` |               |                                                                       |
|         | `true`            | `false`<br>`true` | -                   | `null` | `missing`     | Field required                                                        |

Read the table as three independent rules.

- `required` never looks at a value: it fires on `absent` alone, and only when no `default` was given, which is why
  every `present` and every `null` row ignores it.
- `nullable` never looks at presence: it fires on `null` alone, whether the null came from the source or from a
  `default: null`, which is why it decides nothing in the `present` rows.
- `default` never looks at a value that arrived: it fires on `absent` alone, and once it does, `missing` can no
  longer happen.

Whether a field can be absent at all depends on how it reaches the validator:

| Input Type    | Engine    | How presence is decided                                        |
|---------------|-----------|----------------------------------------------------------------|
| `column`      | both      | Statically, by the DataFrame schema, the same for every row    |
| `struct`      | `pyspark` | Statically, by the struct's field list, the same for every row |
| `dict`        | `pandas`  | Per row, by the keys of the row's dict                         |
| `map`         | `pyspark` | Per row, by the keys of the row's map                          |
| `json_string` | both      | Per row, by the keys of the row's parsed JSON object           |

For `column` and `struct` the field list is fixed for the whole DataFrame, so a field cannot be absent in one row and
present in another. It is the frame that loses the distinction, not the validator: once records are loaded into a
rectangle, a row that never carried a key and a row that carried it as null are the same row. So for a `column` or a
`struct` a default answers only a field the frame's type never carried at all, and a column of nulls, including one
typed `void`, is present, not absent.

#### `required`

The `required` keyword ensures that a property is **present** in the data being validated. It says nothing about the
value: a field the source carried as `null` is present, and is judged by [`nullable`](#nullable) instead.

**JSON Schema**

```json
{
  "type": "object",
  "required": [
    "name"
  ],
  "properties": {
    "name": {
      "type": "string"
    }
  }
}
```

**Python**

```python
from norma.schema import Schema, Column

Schema({
    'name': Column(str, required=True)
})
```

Use the table below to see how `required: ["name"]` behaves:

| Input   | Type             | Output  | Error         | Reason                         |
|---------|------------------|---------|---------------|--------------------------------|
| `"foo"` | `string`         | `"foo"` |               |                                |
| `null`  | `string`         | `null`  | `string_type` | Input should be a valid string |
| `null`  | `[string, null]` | `null`  |               |                                |
|         |                  | `null`  | `missing`     | Field required                 |

What an absence produces depends on where presence is decided:

| Input Type            | Engine    | Not Required              | Required  |
|-----------------------|-----------|---------------------------|-----------|
| `column`, `struct`    | both      | the column is not created | `missing` |
| `dict`, `json_string` | `pandas`  | the column is not created | `missing` |
| `map`, `json_string`  | `pyspark` | `null`                    | `missing` |

For `dict`, `map` and `json_string` the keys are known only when a row is read, so presence is resolved at runtime and
only the rows that lack the key are reported. Whether the field exists at all is decided earlier: `pandas` assembles the
output after reading the data and never creates a field that no row carried, while `pyspark` fixes the output schema
before any value is seen, so every field the schema declares becomes a struct field, `null` in every row that lacked it.

#### `nullable`

Norma has no `nullable` keyword of its own. A property is made nullable by adding `null` to its `type`, as described in
[Null values](#null-values), or by `nullable=True` in the Python API. It governs the **value**, not
the presence: a null is either accepted or reported, while an absent field is never a type error, whatever `nullable`
says.

**JSON Schema**

```json
{
  "type": [
    "string",
    "null"
  ]
}
```

**Python**

```python
from norma.schema import Column

Column(str, nullable=True)
```

Use the table below to see how `nullable` behaves:

| Input   | Type     | Parameters        | Output  | Error         | Reason                                           |
|---------|----------|-------------------|---------|---------------|--------------------------------------------------|
| `"foo"` | `string` | `nullable: true`  | `"foo"` |               |                                                  |
| `null`  | `string` | `nullable: true`  | `null`  |               |                                                  |
| `null`  | `string` | `nullable: false` | `null`  | `string_type` | Input should be a valid string                   |
|         |          | `nullable: false` |         |               | Absence is `required`'s business, not `nullable` |

#### `default`

The `default` keyword is used to specify the value to use when a property is **absent** from the input. It answers
absence alone: a null the source supplied is left as it is. The value is injected before the parsing and the constraints,
so it is coerced and checked like an incoming value.

**JSON Schema**

```json
{
  "type": "string",
  "default": "default_value"
}
```

**Python**

```python
from norma.schema import Column

Column(str, default="default_value")
```

Use the table below to see how `default` behaves:

| Input   | Type      | Parameters                    | Output | Error                | Reason                                                                |
|---------|-----------|-------------------------------|--------|----------------------|-----------------------------------------------------------------------|
|         |           | `default: 25`                 | `25`   |                      |                                                                       |
|         |           | `default: "25"`               | `25`   |                      |                                                                       |
|         |           | `default: "abc"`              | `null` | `int_parsing`        | Input should be a valid integer, unable to parse string as an integer |
|         |           | `default: -5`<br>`minimum: 0` | `null` | `greater_than_equal` | Input should be greater than or equal to 0                            |
|         |           | —                             |        |                      |                                                                       |
|         | `integer` | `default: null`               | `null` | `int_type`           | Input should be a valid integer                                       |
| `null`  | `string`  | `default: "abc"`              | `null` |                      |                                                                       |
| `"abc"` | `integer` | `default: 25`                 | `null` | `int_parsing`        | Input should be a valid integer, unable to parse string as an integer |
| `""`    | `string`  | `default: "n/a"`              | `""`   |                      |                                                                       |

Use a `default` of `null` to guarantee that a column is always there. It is not the same as no default at all: without
one, a column the input never carried never reaches the output, and everything downstream has to guard against its
absence. With it, the column is materialized as null in its declared type whatever the source sent, so the schema, not
the data, decides the shape of the output.

A default fills a slot, and a slot exists only if its container does. A default on a nested field therefore never
creates the object above it: a `null` parent defaults none of its children, and an absent parent is not brought back by
a default on one of them. Giving the parent its own `"default": null` puts it in the output, but as `null`, not as an
object built from its children's defaults.

A `default` that cannot fill a slot is rejected when the schema is built: one on array items, which are never absent,
and a non-null one on an `object` or `array` column.

#### `additionalProperties`

The `additionalProperties` keyword answers the state the other three keywords never see: a field that **is** there but
the schema never declared. It defaults to `false`, so an undeclared field is dropped from the output and reported under
its own path with the `extra_forbidden` error code, whatever the shape it arrived in — a column, a `struct` field, a key
of a`map` or of a Pandas `dict`, or a key of an object held as a JSON string.

**JSON Schema**

```json
{
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "col": {
      "type": "string"
    }
  }
}
```

**Python**

```python
from norma.schema import Schema, Column

Schema({'col': Column(str)}, allow_extra=True)
```

Use the table below to see how `additionalProperties` behaves for an input carrying `foo` and an undeclared `bar`:

| Parameters                    | Output       | Error             | Reason                         |
|-------------------------------|--------------|-------------------|--------------------------------|
| `additionalProperties: true`  | `foo`, `bar` |                   |                                |
| `additionalProperties: false` | `foo`        | `extra_forbidden` | Extra inputs are not permitted |
| —                             | `foo`        | `extra_forbidden` | Extra inputs are not permitted |

### Value rules

#### `enum`

The `enum` keyword is used to restrict a property to a predefined set of values.

**JSON Schema**

```json
{
  "type": "string",
  "enum": [
    "red",
    "green",
    "blue"
  ]
}
```

**Python**

```python
from norma.schema import Column

Column(str, isin=["red", "green", "blue"])
```

Use the table below to see how `enum: ["red","green","blue"]` behaves:

| Input      | Type                                                                | Output | Error  | Reason                                   |
|------------|---------------------------------------------------------------------|--------|--------|------------------------------------------|
| `null`     | `number`<br>`integer`<br>`string`<br>`bool`<br>`datetime`<br>`date` | `null` |        |                                          |
| `"red"`    | `string`                                                            |        |        |
| `"yellow"` | `string`                                                            | `null` | `enum` | Input should be "red", "green" or "blue" |

#### `const`

The `const` keyword is used to restrict a property to a single specific value.

**JSON Schema**

```json
{
  "type": "string",
  "const": "fixed_value"
}
```

**Python**

```python
from norma.schema import Column

Column(str, eq="fixed_value")
```

Use the table below to see how `const: "foo"` behaves:

| Input   | Type                                                                | Output  | Error      | Reason                         |
|---------|---------------------------------------------------------------------|---------|------------|--------------------------------|
| `null`  | `number`<br>`integer`<br>`string`<br>`bool`<br>`datetime`<br>`date` | `null`  |            |                                |
| `"foo"` | `string`                                                            | `"foo"` |            |                                |
| `"bar"` | `string`                                                            | `null`  | `equal_to` | Input should be equal to "foo" |

#### `not`

The `not` keyword is used to specify that a property must not match a given rule.
Norma currently supports `not` with `enum` and `const` rules.

##### `not enum`

**JSON Schema**

```json
{
  "type": "string",
  "not": {
    "enum": [
      "red",
      "green",
      "blue"
    ]
  }
}
```

**Python**

```python
from norma.schema import Column

Column(str, notin=["red", "green", "blue"])
```

Use the table below to see how `not enum: ["red","green","blue"]` behaves:

| Input   | Type                                                                | Output  | Error      | Reason                                       |
|---------|---------------------------------------------------------------------|---------|------------|----------------------------------------------|
| `null`  | `number`<br>`integer`<br>`string`<br>`bool`<br>`datetime`<br>`date` | `null`  |            |                                              |
| `"abc"` | `string`                                                            | `"abc"` |            |                                              |
| `"red"` | `string`                                                            | `null`  | `not_enum` | Input should not be "red", "green" or "blue" |

##### `not const`

**JSON Schema**

```json
{
  "type": "string",
  "not": {
    "const": "fixed_value"
  }
}
```

**Python**

```python
from norma.schema import Column

Column(str, ne="fixed_value")
```

Use the table below to see how `not const: "bar"` behaves:

| Input   | Type                                                                | Output  | Error          | Reason                             |
|---------|---------------------------------------------------------------------|---------|----------------|------------------------------------|
| `null`  | `number`<br>`integer`<br>`string`<br>`bool`<br>`datetime`<br>`date` | `null`  |                |                                    |
| `"foo"` | `string`                                                            | `"foo"` |                |                                    |
| `"bar"` | `string`                                                            | `null`  | `not_equal_to` | Input should not be equal to "bar" |

#### `minimum`

The `minimum` keyword is used to specify the minimum allowable value for given types.

**JSON Schema**

```json
{
  "type": "number",
  "minimum": 0
}
```

**Python**

```python
from norma.schema import Column

Column(float, ge=0)
```

Use the table below to see how `minimum` behaves:

| Input                    | Type                                                                | Parameters                        | Output                   | Error                | Reason                                                          |
|--------------------------|---------------------------------------------------------------------|-----------------------------------|--------------------------|----------------------|-----------------------------------------------------------------|
| `null`                   | `number`<br>`integer`<br>`string`<br>`bool`<br>`datetime`<br>`date` |                                   | `null`                   |                      |                                                                 |
| `3`                      | `integer`                                                           | `minimum: 3`                      | `3`                      |                      |                                                                 |
| `2`                      | `integer`                                                           | `minimum: 3`                      | `null`                   | `greater_than_equal` | Input should be greater than or equal to 3                      |
| `3.5`                    | `number`                                                            | `minimum: 3.5`                    | `3.5`                    |                      |                                                                 |
| `2.5`                    | `number`                                                            | `minimum: 3.5`                    | `null`                   | `greater_than_equal` | Input should be greater than or equal to 3.5                    |
| `"c"`                    | `string`                                                            | `minimum: "c"`                    | `"c"`                    |                      |                                                                 |
| `"b"`                    | `string`                                                            | `minimum: "c"`                    | `null`                   | `greater_than_equal` | Input should be greater than or equal to "c"                    |
| `true`                   | `boolean`                                                           | `minimum: true`                   | `true`                   |                      |                                                                 |
| `false`                  | `boolean`                                                           | `minimum: true`                   | `null`                   | `greater_than_equal` | Input should be greater than or equal to true                   |
| `"2023-01-02"`           | `date`                                                              | `minimum: "2023-01-02"`           | `"2023-01-02"`           |                      |                                                                 |
| `"2023-01-01"`           | `date`                                                              | `minimum: "2023-01-02"`           | `null`                   | `greater_than_equal` | Input should be greater than or equal to "2023-01-02"           |
| `"2023-01-02T00:00:00Z"` | `datetime`                                                          | `minimum: "2023-01-02T00:00:00Z"` | `"2023-01-02T00:00:00Z"` |                      |                                                                 |
| `"2023-01-01T23:59:59Z"` | `datetime`                                                          | `minimum: "2023-01-02T00:00:00Z"` | `null`                   | `greater_than_equal` | Input should be greater than or equal to "2023-01-02T00:00:00Z" |

#### `exclusiveMinimum`

The `exclusiveMinimum` keyword is used to specify that the value must be greater than (but not equal to) a given value.

**JSON Schema**

```json
{
  "type": "number",
  "exclusiveMinimum": 0
}
```

**Python**

```python
from norma.schema import Column

Column(float, gt=0)
```

Norma also accepts the draft-4 form, where `exclusiveMinimum` is a boolean that modifies a neighbouring `minimum`
instead of carrying the bound itself:

```json
{
  "type": "number",
  "minimum": 3,
  "exclusiveMinimum": true
}
```

`true` turns the bound into `greater_than(gt=3)`, `false` leaves it as the inclusive `greater_than_equal(ge=3)`.
When `exclusiveMinimum` sits next to a `minimum` but is not a boolean, `Schema.from_json_schema` raises `ValueError`.

Use the table below to see how `exclusiveMinimum` behaves:

| Input                    | Type                                                                | Parameters                                 | Output                   | Error          | Reason                                              |
|--------------------------|---------------------------------------------------------------------|--------------------------------------------|--------------------------|----------------|-----------------------------------------------------|
| `null`                   | `number`<br>`integer`<br>`string`<br>`bool`<br>`datetime`<br>`date` |                                            | `null`                   |                |                                                     |
| `4`                      | `integer`                                                           | `exclusiveMinimum: 3`                      | `4`                      |                |                                                     |
| `3`                      | `integer`                                                           | `exclusiveMinimum: 3`                      | `null`                   | `greater_than` | Input should be greater than 3                      |
| `4.5`                    | `number`                                                            | `exclusiveMinimum: 3.5`                    | `4.5`                    |                |                                                     |
| `3.5`                    | `number`                                                            | `exclusiveMinimum: 3.5`                    | `null`                   | `greater_than` | Input should be greater than 3.5                    |
| `"d"`                    | `string`                                                            | `exclusiveMinimum: "c"`                    | `"d"`                    |                |                                                     |
| `"c"`                    | `string`                                                            | `exclusiveMinimum: "c"`                    | `null`                   | `greater_than` | Input should be greater than "c"                    |
| `true`                   | `boolean`                                                           | `exclusiveMinimum: false`                  | `true`                   |                |                                                     |
| `true`                   | `boolean`                                                           | `exclusiveMinimum: true`                   | `null`                   | `greater_than` | Input should be greater than true                   |
| `"2023-01-03"`           | `date`                                                              | `exclusiveMinimum: "2023-01-02"`           | `"2023-01-03"`           |                |                                                     |
| `"2023-01-02"`           | `date`                                                              | `exclusiveMinimum: "2023-01-02"`           | `null`                   | `greater_than` | Input should be greater than "2023-01-02"           |
| `"2023-01-03T00:00:00Z"` | `datetime`                                                          | `exclusiveMinimum: "2023-01-02T00:00:00Z"` | `"2023-01-03T00:00:00Z"` |                |                                                     |
| `"2023-01-02T00:00:00Z"` | `datetime`                                                          | `exclusiveMinimum: "2023-01-02T00:00:00Z"` | `null`                   | `greater_than` | Input should be greater than "2023-01-02T00:00:00Z" |

#### `maximum`

The `maximum` keyword is used to specify the maximum allowable value for given types.

**JSON Schema**

```json
{
  "type": "number",
  "maximum": 100
}
```

**Python**

```python
from norma.schema import Column

Column(float, le=100)
```

Use the table below to see how `maximum` behaves:

| Input                    | Type                                                                | Parameters                        | Output                   | Error             | Reason                                                       |
|--------------------------|---------------------------------------------------------------------|-----------------------------------|--------------------------|-------------------|--------------------------------------------------------------|
| `null`                   | `number`<br>`integer`<br>`string`<br>`bool`<br>`datetime`<br>`date` |                                   | `null`                   |                   |                                                              |
| `3`                      | `integer`                                                           | `maximum: 3`                      | `3`                      |                   |                                                              |
| `4`                      | `integer`                                                           | `maximum: 3`                      | `null`                   | `less_than_equal` | Input should be less than or equal to 3                      |
| `3.5`                    | `number`                                                            | `maximum: 3.5`                    | `3.5`                    |                   |                                                              |
| `4.0`                    | `number`                                                            | `maximum: 3.5`                    | `null`                   | `less_than_equal` | Input should be less than or equal to 3.5                    |
| `"c"`                    | `string`                                                            | `maximum: "c"`                    | `"c"`                    |                   |                                                              |
| `"d"`                    | `string`                                                            | `maximum: "c"`                    | `null`                   | `less_than_equal` | Input should be less than or equal to "c"                    |
| `true`                   | `boolean`                                                           | `maximum: true`                   | `true`                   |                   |                                                              |
| `true`                   | `boolean`                                                           | `maximum: false`                  | `null`                   | `less_than_equal` | Input should be less than or equal to false                  |
| `"2023-01-02"`           | `date`                                                              | `maximum: "2023-01-02"`           | `"2023-01-02"`           |                   |                                                              |
| `"2023-01-03"`           | `date`                                                              | `maximum: "2023-01-02"`           | `null`                   | `less_than_equal` | Input should be less than or equal to "2023-01-02"           |
| `"2023-01-02T00:00:00Z"` | `datetime`                                                          | `maximum: "2023-01-02T00:00:00Z"` | `"2023-01-02T00:00:00Z"` |                   |                                                              |
| `"2023-01-02T12:34:56Z"` | `datetime`                                                          | `maximum: "2023-01-02T00:00:00Z"` | `null`                   | `less_than_equal` | Input should be less than or equal to "2023-01-02T00:00:00Z" |

#### `exclusiveMaximum`

The `exclusiveMaximum` keyword is used to specify that the value must be less than (but not equal to) a given value.

**JSON Schema**

```json
{
  "type": "number",
  "exclusiveMaximum": 100
}
```

**Python**

```python
from norma.schema import Column

Column(float, lt=100)
```

Norma also accepts the draft-4 form, where `exclusiveMaximum` is a boolean that modifies a neighbouring `maximum`
instead of carrying the bound itself:

```json
{
  "type": "number",
  "maximum": 3,
  "exclusiveMaximum": true
}
```

`true` turns the bound into `less_than(lt=3)`, `false` leaves it as the inclusive `less_than_equal(le=3)`.
When `exclusiveMaximum` sits next to a `maximum` but is not a boolean, `Schema.from_json_schema` raises `ValueError`.

Use the table below to see how `exclusiveMaximum` behaves:

| Input                    | Type                                                                | Parameters                                 | Output                   | Error       | Reason                                           |
|--------------------------|---------------------------------------------------------------------|--------------------------------------------|--------------------------|-------------|--------------------------------------------------|
| `null`                   | `number`<br>`integer`<br>`string`<br>`bool`<br>`datetime`<br>`date` |                                            | `null`                   |             |                                                  |
| `2`                      | `integer`                                                           | `exclusiveMaximum: 3`                      | `2`                      |             |                                                  |
| `3`                      | `integer`                                                           | `exclusiveMaximum: 3`                      | `null`                   | `less_than` | Input should be less than 3                      |
| `2.5`                    | `number`                                                            | `exclusiveMaximum: 3.5`                    | `2.5`                    |             |                                                  |
| `3.5`                    | `number`                                                            | `exclusiveMaximum: 3.5`                    | `null`                   | `less_than` | Input should be less than 3.5                    |
| `"b"`                    | `string`                                                            | `exclusiveMaximum: "c"`                    | `"b"`                    |             |                                                  |
| `"c"`                    | `string`                                                            | `exclusiveMaximum: "c"`                    | `null`                   | `less_than` | Input should be less than "c"                    |
| `false`                  | `boolean`                                                           | `exclusiveMaximum: true`                   | `false`                  |             |                                                  |
| `true`                   | `boolean`                                                           | `exclusiveMaximum: true`                   | `null`                   | `less_than` | Input should be less than true                   |
| `"2023-01-01"`           | `date`                                                              | `exclusiveMaximum: "2023-01-02"`           | `"2023-01-01"`           |             |                                                  |
| `"2023-01-02"`           | `date`                                                              | `exclusiveMaximum: "2023-01-02"`           | `null`                   | `less_than` | Input should be less than "2023-01-02"           |
| `"2023-01-01T00:00:00Z"` | `datetime`                                                          | `exclusiveMaximum: "2023-01-02T00:00:00Z"` | `"2023-01-01T00:00:00Z"` |             |                                                  |
| `"2023-01-02T00:00:00Z"` | `datetime`                                                          | `exclusiveMaximum: "2023-01-02T00:00:00Z"` | `null`                   | `less_than` | Input should be less than "2023-01-02T00:00:00Z" |

#### `minLength`

The `minLength` keyword ensures that string values have at least the specified number of characters.

**JSON Schema**

```json
{
  "type": "string",
  "minLength": 3
}
```

**Python**

```python
from norma.schema import Column

Column(str, min_length=3)
```

Use the table below to see how `minLength: 3` behaves:

| Input   | Type     | Output  | Error              | Reason                                   |
|---------|----------|---------|--------------------|------------------------------------------|
| `null`  | `string` | `null`  |                    |                                          |
| `"abc"` | `string` | `"abc"` |                    |                                          |
| `"ab"`  | `string` | `null`  | `string_too_short` | String should have at least 3 characters |

#### `maxLength`

The `maxLength` keyword ensures that string values do not exceed the specified number of characters.

**JSON Schema**

```json
{
  "type": "string",
  "maxLength": 10
}
```

**Python**

```python
from norma.schema import Column

Column(str, max_length=10)
```

Use the table below to see how `maxLength: 3` behaves:

| Input    | Type     | Output  | Error             | Reason                                  |
|----------|----------|---------|-------------------|-----------------------------------------|
| `null`   | `string` | `null`  |                   |                                         |
| `"abc"`  | `string` | `"abc"` |                   |                                         |
| `"abcd"` | `string` | `null`  | `string_too_long` | String should have at most 3 characters |

#### `pattern`

The `pattern` keyword constrains string values to match the provided regular expression.

**JSON Schema**

```json
{
  "type": "string",
  "pattern": "^[A-Z]{2}\\d{4}$"
}
```

**Python**

```python
from norma.schema import Column

Column(str, pattern=r'^[A-Z]{2}\d{4}$')
```

Use the table below to see how `pattern: "^[a-z]*$` behaves:

| Input    | Type     | Output  | Error                     | Reason                                 |
|----------|----------|---------|---------------------------|----------------------------------------|
| `null`   | `string` | `null`  |                           |                                        |
| `"abc"`  | `string` | `"abc"` |                           |                                        |
| `"abc1"` | `string` | `null`  | `string_pattern_mismatch` | String should match pattern "^[a-z]*$" |

#### `multipleOf`

The `multipleOf` keyword ensures that numeric values are exact multiples of the specified divisor.

**JSON Schema**

```json
{
  "type": "number",
  "multipleOf": 0.5
}
```

**Python**

```python
from norma.schema import Column

Column(float, multiple_of=0.5)
```

Use the table below to see how `multipleOf: 10` behaves:

| Input          | Type                  | Output | Error         | Reason                           |
|----------------|-----------------------|--------|---------------|----------------------------------|
| `null`         | `number`<br>`integer` | `null` |               |                                  |
| `30.0`<br>`30` | `number`<br>`integer` | `30`   |               |                                  |
| `33.0`<br>`33` | `number`<br>`integer` | `null` | `multiple_of` | Input should be a multiple of 10 |

#### `minItems`

The `minItems` keyword ensures that array values contain at least the specified number of elements.

**JSON Schema**

```json
{
  "type": "array",
  "items": {
    "type": "string"
  },
  "minItems": 1
}
```

**Python**

```python
from norma.schema import Schema, Column

Schema({
    'tags': Column(list, inner_schema=Column(str), min_items=1)
})
```

Use the table below to see how `minItems: 3` behaves:

| Input              | Type            | Output             | Error       | Reason                             |
|--------------------|-----------------|--------------------|-------------|------------------------------------|
| `["ab","cd","ef"]` | `array<string>` | `["ab","cd","ef"]` |             |                                    |
| `["ab","cd"]`      | `array<string>` | `null`             | `too_short` | Array should have at least 3 items |

#### `maxItems`

The `maxItems` keyword ensures that array values contain no more than the specified number of elements.

**JSON Schema**

```json
{
  "type": "array",
  "items": {
    "type": "string"
  },
  "maxItems": 5
}
```

**Python**

```python
from norma.schema import Schema, Column

Schema({
    'tags': Column(list, inner_schema=Column(str), max_items=5)
})
```

Use the table below to see how `maxItems: 2` behaves:

| Input              | Type            | Output        | Error      | Reason                            |
|--------------------|-----------------|---------------|------------|-----------------------------------|
| `["ab","cd"]`      | `array<string>` | `["ab","cd"]` |            |                                   |
| `["ab","cd","ef"]` | `array<string>` | `null`        | `too_long` | Array should have at most 2 items |

#### `uniqueItems`

The `uniqueItems` keyword ensures that array values do not contain duplicate elements.

**JSON Schema**

```json
{
  "type": "array",
  "items": {
    "type": "string"
  },
  "uniqueItems": true
}
```

**Python**

```python
from norma.schema import Schema, Column

Schema({
    'tags': Column(list, inner_schema=Column(str), unique_items=True)
})
```

Use the table below to see how `uniqueItems` behaves:

| Input                       | Type            | Output                      | Error          | Reason                                   |
|-----------------------------|-----------------|-----------------------------|----------------|------------------------------------------|
| `["ab","cd","ef"]`          | `array<string>` | `["ab","cd","ef"]`          |                |                                          |
| `["ab","cd","ab"]`          | `array<string>` | `null`                      | `unique_items` | Input should not contain duplicate items |
| `[{"v":1},{"v":2},{"v":3}]` | `array<object>` | `[{"v":1},{"v":2},{"v":3}]` |                |                                          |
| `[{"v":1},{"v":2},{"v":1}]` | `array<object>` | `null`                      | `unique_items` | Input should not contain duplicate items |
