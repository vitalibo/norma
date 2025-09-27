# Engine Support

Norma ships with engine-specific validators that apply schemas against DataFrame workloads. At runtime,
`norma.schema.Schema.validate` inspects the input object and dispatches to the matching engine adapter. This page
outlines the supported engines and any notable capabilities or limitations.

## Pandas

- **How it works**: When the input is a `pandas.DataFrame`, Norma uses `norma.engines.pandas.validator.validate`. Each
  column rule is resolved into a Pandas-aware implementation (see `norma.engines.pandas.rules`).
- **Error handling**: Errors are collected in an engine-specific `norma.engines.pandas.rules.ErrorState` and returned in
  the requested `error_col`. The validator keeps the original column values alongside normalised ones for easier
  debugging.
- **Nested data**: Currently, nested data structures (e.g., lists or dictionaries within cells) are not supported.
- **Dependencies**: Requires `pandas` (and `numpy`, which Pandas brings along). User code should import and pass a
  Pandas DataFrame; no additional setup is needed.

### Example

```python
import pandas as pd

from norma.schema import Column, Schema

schema = Schema({
    'email': Column(str, pattern=r'^\S+@\S+$'),
    'age': Column(int, ge=18, default=18),
})

df = pd.DataFrame({
    'email': ['user@example.com'],
    'age': [-1]
})

df = schema.validate(df)
```

## Pyspark

- **How it works**: When given a `pyspark.sql.DataFrame`, Norma executes `norma.engines.pyspark.validator.validate`.
  Rule proxies are turned into Spark column expressions defined in `norma.engines.pyspark.rules`.
- **Error handling**: Errors are accumulated in a struct-array column managed by the engine
  `norma.engines.pyyspark.rules.ErrorState`. Each failure stores the rule metadata and, when validating arrays, the
  failing element indexes.
- **Nested data**: Schema nesting is supported for structs and arrays of structs. Deeply nested arrays (arrays of
  arrays) are not yet implemented.
- **Dependencies**: Requires `pyspark`. Spark sessions must be initialised by user code before calling
  `norma.schema.Schema.validate`.

### Example

```python
from pyspark.sql import SparkSession

from norma.schema import Column, Schema

spark = SparkSession.builder.getOrCreate()

schema = Schema({
    'email': Column(str, pattern=r'^\S+@\S+$'),
    'age': Column(int, ge=18, default=18),
})

df = spark.createDataFrame([
    ('user@example.com', -1)
], schema=['email', 'age'])

df = schema.validate(df)
```

## Custom Engines

Norma can be extended with additional engines by implementing equivalents of `validator.validate`, engine-specific rule
translations, and an `ErrorState` that conforms to `norma.rules.ErrorState`. New engines should mirror the Pandas
and Pyspark adapters to remain compatible with the high-level schema API.
