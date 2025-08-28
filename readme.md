# Norma

![status](https://github.com/vitalibo/norma/actions/workflows/ci.yaml/badge.svg)

Norma is a data validation framework designed for Pandas and PySpark DataFrames, leveraging JSON Schema or a Python API
to enforce data integrity.
By applying predefined validation rules, it systematically identifies inconsistencies without disrupting the data
pipeline.
Instead of halting execution upon encountering invalid data, Norma introduces a dedicated errors column to capture
validation details while resetting erroneous values to null.
This approach ensures a resilient, declarative, and non-intrusive method for maintaining data quality at scale.

### Install

Install with `pip`:

```bash
pip install 'git+https://github.com/vitalibo/norma.git'
```

### Quick Start

First, let's create a validation Schema.
To do this, we can use a JSON Schema.

```json
{
  "$id": "https://example.com/address.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "minLength": 2
    },
    "age": {
      "type": "integer",
      "minimum": 0,
      "maximum": 120
    },
    "sex": {
      "type": "string",
      "enum": [
        "M",
        "F"
      ]
    },
    "email": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
    }
  },
  "required": [
    "name",
    "age",
    "email"
  ]
}
```

```python
from norma.schema import Schema

schema = Schema.from_json_schema({... see above ...})
```

or, we can use a Python API.

```python
from norma.schema import Column, Schema
from norma import rules

schema = Schema({
    'name': Column(str, min_length=2),
    'age': Column(int, ge=0, le=120),
    'sex': Column(str, rules=rules.isin(['M', 'F'])),
    'email': Column(str, pattern=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
})
```

Now, we are ready to validate our data.
Let's create a Pandas DataFrame with some invalid data.

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['John', 'Rian', 'Alice'],
    'age': [42, 130, 25],
    'sex': ['M', 'F', 'X'],
    'email': ['john@spring.ep', 'ryna@spring.ep', 'alice.spring@ep']
})
```

or same for PySpark DataFrame.

```python
from pyspark.sql import SparkSession

spark_session = SparkSession.builder.getOrCreate()

df = spark_session.createDataFrame([
    ('John', 42, 'M', 'john@spring.ep'),
    ('Rian', 130, 'F', 'ryna@spring.ep'),
    ('Alice', 25, 'X', 'alice.spring@ep')
], ['name', 'age', 'sex', 'email'])
```

And validate it.

```python
actual = schema.validate(df)
```

Output:

For Pandas DataFrame we have:

```text
    name   age   sex           email  errors
0   John    42     M  john@spring.ep  {}
1   Rian  <NA>     F  ryna@spring.ep  {'age': {'details': [{'type': 'less_than_equal', 'msg': 'Input should be less than or equal to 120'}], 'original': '130'}}
2  Alice    25  <NA>            <NA>  {'sex': {'details': [{'type': 'enum', 'msg': 'Input should be "M" or "F"'}], 'original': '"X"'}, 'email': {'details': [{'type': 'string_pattern_mismatch', 'msg': 'String should match pattern "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"'}], 'original': '"alice.spring@ep"'}}
```

and dtypes:

```text
name      string[python]
age                Int64
sex       string[python]
email     string[python]
errors            object
dtype: object
```

Similarly, for PySpark DataFrame:

```text
+-----+----+----+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|name |age |sex |email         |errors                                                                                                                                                                                         |
+-----+----+----+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|John |42  |M   |john@spring.ep|{}                                                                                                                                                                                             |
|Rian |NULL|F   |ryna@spring.ep|{age -> {[{less_than_equal, Input should be less than or equal to 120}], 130}}                                                                                                                 |
|Alice|25  |NULL|NULL          |{email -> {[{string_pattern_mismatch, String should match pattern "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"}], "alice.spring@ep"}, sex -> {[{enum, Input should be "M" or "F"}], "X"}}|
+-----+----+----+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

and schema:

```text
root
 |-- name: string (nullable = true)
 |-- age: integer (nullable = true)
 |-- sex: string (nullable = true)
 |-- email: string (nullable = true)
 |-- errors: map (nullable = false)
 |    |-- key: string
 |    |-- value: struct (valueContainsNull = true)
 |    |    |-- details: array (nullable = false)
 |    |    |    |-- element: struct (containsNull = true)
 |    |    |    |    |-- type: string (nullable = true)
 |    |    |    |    |-- msg: string (nullable = true)
 |    |    |-- original: string (nullable = true)
```

### Supported validation rules

Norma supports a variety of validation rules, including:

| rule                 | pandas | pyspark | pyspark[object] | pyspark[array] |
|----------------------|--------|---------|-----------------|----------------|
| `required`           | ✅      | ✅       | ✅               | ✅              |
| `equal_to`           | ✅      | ✅       | ✅               | ✅              |
| `not_equal_to`       | ✅      | ✅       | ✅               | ✅              |
| `greater_than`       | ✅      | ✅       | ✅               | ✅              |
| `greater_than_equal` | ✅      | ✅       | ✅               | ✅              |
| `less_than`          | ✅      | ✅       | ✅               | ✅              |
| `less_than_equal`    | ✅      | ✅       | ✅               | ✅              |
| `multiple_of`        | ✅      | ✅       | ✅               | ✅              |
| `min_length`         | ✅      | ✅       | ✅               | ✅              |
| `max_length`         | ✅      | ✅       | ✅               | ✅              |
| `pattern`            | ✅      | ✅       | ✅               | ✅              |
| `isin`               | ✅      | ✅       | ✅               | ✅              |
| `notin`              | ✅      | ✅       | ✅               | ✅              |
| `extra_forbidden`    | ✅      | ✅       | ✅               | ➖              |
| `unique_items`       | ➖      | ➖       | ➖               | ✅              |
| `int_parsing`        | ✅      | ✅       | ✅               | ✅              |
| `float_parsing`      | ✅      | ✅       | ✅               | ✅              |
| `str_parsing`        | ✅      | ✅       | ✅               | ✅              |
| `bool_parsing`       | ✅      | ✅       | ✅               | ✅              |
| `date_parsing`       | ✅      | ✅       | ✅               | ✅              |
| `time_parsing`       | ✅      | ✅       | ✅               | ✅              |
| `datetime_parsing`   | ✅      | ✅       | ✅               | ✅              |
| `duration_parsing`   | ✅      | ✅       | ✅               | ✅              |
| `uuid_parsing`       | ✅      | ✅       | ✅               | ✅              |
| `ipv4_address`       | ✅      | ✅       | ✅               | ✅              |
| `ipv6_address`       | ✅      | ✅       | ✅               | ✅              |
| `uri_parsing`        | ✅      | ✅       | ✅               | ✅              |
| `object_parsing`     | ❌      | ✅       | ✅               | ✅              |
| `array_parsing`      | ❌      | ✅       | ✅               | ❌              |
### Errors

The error format is a dictionary where the key is the column name, and value is a structure with two fields: `details`
and `original`. The `details` field is a list of details about the error, and the `original` field is the original value
that caused the error.

```json
{
  "name": "Alice",
  "age": 25,
  "sex": null,
  "email": null,
  "errors": {
    "sex": {
      "details": [
        {
          "type": "enum",
          "msg": "Input should be \"M\" or \"F\""
        }
      ],
      "original": "\"X\""
    },
    "email": {
      "details": [
        {
          "type": "string_pattern_mismatch",
          "msg": "String should match pattern \"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$\""
        }
      ],
      "original": "\"alice.spring@ep\""
    }
  }
}
```

Special case for PySpark, when DataFrame has array validation rules, the error format is slightly different.
In this case, the `details` array has an additional `loc` field that indicates the indexes of the array elements that
failed validation and incorrect values are replaced with `null` in an array.

```json
{
  "tags": [
    null,
    "tag1",
    null
  ],
  "errors": {
    "tags[]": {
      "details": [
        {
          "loc": [ 0, 2 ],
          "type": "enum",
          "msg": "Input should be \"tag1\", \"tag2\" or \"tag3\""
        }
      ],
      "original": "[\"tag0\",\"tag1\",\"tag4\"]"
    }
  }
}
```
