# Rules

Norma's rules module defines the interfaces that power validation as well as a collection of factory helpers that create
engine-specific rule proxies. These proxies are resolved by the active engine (Pandas or Pyspark) during validation.

## ErrorState

Abstract base class responsible for accumulating and persisting validation errors.

### Methods

#### add_errors

Store error details for values where the boolean mask is `True`. Engine implementations decide how errors are
persisted (e.g. appended to a column or aggregated in-place).

##### Parameters

- `boolmask`: Boolean expression that indicates which rows failed validation.
- `column`: The name of the column where the errors occurred.
- `details`: Dictionary that provides error `type` and `msg` details.

## Rule

Interface for all validation rules.

### Methods

#### verify

Core hook invoked by validators.

##### Parameters

- `df`: Input `DataFrame`.
- `column`: The name of the column being validated.
- `error_state`: An instance of `norma.rules.ErrorState` used to record validation errors.

##### Returns

- Optionally returns a transformed `DataFrame`.

## Built-in Rule Factories

Norma provides a set of built-in rule factories that create `RuleProxy` instances.
These factories cover common validation scenarios.

### Presence

- `required()` — reject null or missing values.
- `extra_forbidden(allowed)` — reject columns not explicitly allowed.

### Equality

- `equal_to(eq)` — ensure values equal `eq`.
- `not_equal_to(ne)` — ensure values differ from `ne`.

### Comparisons

- `greater_than(gt)` - restrict values to be greater than `gt`.
- `greater_than_equal(ge)`- restrict values to be greater than or equal to `ge`.
- `less_than(lt)` - restrict values to be less than `lt`.
- `less_than_equal(le)` - restrict values to be less than or equal to `le`.

### Numeric Shape

- `multiple_of(multiple)` — ensure values are multiples of `multiple`.

### String Shape

- `min_length(value)` — minimum length for strings.
- `max_length(value)` — maximum length for strings.
- `pattern(regex)` — ensure values match a compiled regular expression.

### Membership

- `isin(values)` — restrict to a whitelist of values.
- `notin(values)` — reject values found in the provided iterable.

### Collection

- `min_items(value)` — minimum item count for lists.
- `max_items(value)` — maximum item count for lists.
- `unique_items()` — enforce uniqueness within array elements.

### Type Coercion

All parsing helpers run before constraint checks and attempt to coerce data into the expected type. When an engine
cannot perform a true cast, the helper validates the format and leaves the value as a string.

- `int_parsing()` - coerce values to integers.
- `float_parsing()` - coerce values to floats.
- `str_parsing()` - coerce values to strings.
- `bool_parsing()` - coerce values to booleans.
- `datetime_parsing()` - coerce values to datetime objects.
- `date_parsing()` - coerce values to date objects.
- `time_parsing()` - coerce values to time objects.
- `duration_parsing()` - coerce values to duration/timedelta objects.
- `uuid_parsing()` - coerce values to UUID objects (Pandas returns normalized UUID strings).
- `ipv4_address()` - validate IPv4 strings and return the original string value.
- `ipv6_address()` - validate IPv6 strings and return the original string value.
- `uri_parsing()` - validate URI strings and return the original string value.
- `object_parsing(schema)` — coerce nested objects using the supplied schema (Pyspark only; Pandas raises exception).
- `array_parsing(schema)` — coerce arrays/lists using the supplied item schema (Pyspark only; Pandas raises exception).

## Custom Rules

Users can implement custom rules by subclassing `Rule` and implementing the `verify` method.

### Custom Pandas Rules

Users have two ways to create custom Pandas rules:

```python
from norma.engines.pandas.rules import rule
from norma.schema import Schema, Column

schema = Schema({
    "even": Column(int, rules=rule(
        lambda df, col: df[col][df[col].notna()] % 2 != 0,
        details={
            "type": "even_number",
            "msg": "Value must be an even number"
        }
    ))
})
```

or, by subclassing `norma.rules.Rule`:

```python
from norma.rules import Rule
from norma.schema import Schema, Column


class MyRule(Rule):
    def verify(self, df, column, error_state):
        mask =  df[column][df[column].notna()] % 2 != 0
        error_state.add_errors(
            mask, column,
            details={
                "type": "even_number",
                "msg": "Value must be an even number"
            }
        )
        return df[column]


schema = Schema({
    "even": Column(int, rules=MyRule())
})
```

During validation, the `verify` method receives the entire `pd.DataFrame`, the name of the column being validated,
and an instance of `norma.rules.ErrorState` to record validation errors. The method should return the (optionally
transformed) column.

### Custom Pyspark Rules

Like Pandas, users can create custom Pyspark rules using a factory function or by subclassing `norma.rules.Rule`.

```python
from norma.engines.pyspark.rules import rule
from norma.schema import Schema, Column

from pyspark.sql import functions as fn

schema = Schema({
    "even": Column(int, rules=rule(
        lambda col_expr: (col_expr % fn.lit(2)) != fn.lit(0),
        details={
            "type": "even_number",
            "msg": "Value must be an even number"
        }
    ))
})
```

or, by subclassing `norma.rules.Rule`:

```python
from norma.rules import Rule
from norma.schema import Schema, Column
from norma.engines.pyspark.utils import flatten_nested_values

from pyspark.sql import functions as fn


class MyRule(Rule):
    def verify(self, df, column, error_state):
        def func(col_expr):
            return (col_expr % fn.lit(2)) != fn.lit(0)

        details = {
            "type": "even_number",
            "msg": "Value must be an even number"
        }

        if "[]" not in column:
            return df.transform(
                error_state.add_errors(func(fn.col(column)), column, details=details))

        # handle cases where the column is an array
        return df.transform(
            error_state.add_errors(
                fn.transform(flatten_nested_values(column), func), column, details=details))


schema = Schema({
    "even": Column(int, rules=MyRule())
})
```

During validation, the `verify` method receives the entire `pyspark.sql.DataFrame`, the name of the column being
validated, and an instance of `norma.rules.ErrorState` to record validation errors. The method should return transformed
dataframe. Unlike Pandas, Pyspark supports nested columns and arrays, so when implementing custom rules, users should
account for these scenarios.
