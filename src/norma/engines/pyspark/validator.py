from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.types import StructType

import norma.rules
from norma.engines.pyspark.rules import ErrorState, default, extra_forbidden
from norma.engines.pyspark.utils import dtype_at, format_value, materialize, nested_get_expr, nested_set_expr

if TYPE_CHECKING:
    from norma.schema import Schema


def validate(schema: Schema, df: DataFrame, error_column: str) -> DataFrame:
    """
    Validate the PySpark DataFrame according to the schema

    :param schema: The schema to validate the DataFrame against
    :param df: The DataFrame to validate
    :param error_column: The name of the column to store error information
    """

    original_cols = df.columns
    error_state = ErrorState(df, error_column, with_loc=schema.contains_arrays)

    df = _validate(df, schema, error_state, original_cols)
    df = error_state.flush(df)

    df = _format_error_details(df, error_state)
    df = _nullify_invalid_values(df, schema, error_state)

    kept_cols = [name for name in schema.columns if name in df.columns]
    if schema.allow_extra:
        kept_cols = [*original_cols, *(col for col in kept_cols if col not in original_cols)]
    return df.select(*kept_cols, error_column)


def _validate(df, schema, error_state, original_cols, parent=''):
    """
    Recursively validate the DataFrame according to the schema
    """

    names = {
        name for name, column in schema.columns.items()
        if name in original_cols or column.required or column.has_default
    }
    if not schema.allow_extra:
        names.update(original_cols)

    for name in sorted(names):
        column = f'{parent}{name}'
        error_state.stub_if_absent(column)
        if name not in schema.columns:
            df = _apply_rules(df, [extra_forbidden()], column, error_state)
            continue

        definition = schema.columns[name]
        rules = [default(definition.default)] if definition.has_default else []
        df = _apply_rules(df, [*rules, *definition.rules], column, error_state)

        # if there is inner schema, validate recursively
        inner_schema = definition.inner_schema
        if inner_schema is None:
            continue

        if definition.dtype in {'array', 'list'}:
            column = f'{column}[]'
            df = _apply_rules(df, inner_schema.rules, column, error_state)
            inner_schema = inner_schema.inner_schema

        dtype = error_state.dtype_of(column)
        if not isinstance(dtype, StructType):
            continue

        df = _validate(df, inner_schema, error_state, dtype.fieldNames(), f'{column}.')
        if not error_state.dtype_of(column).fields:
            error_state.drop_column(column.removesuffix('[]'))

    return df


def _apply_rules(df, rules, column, error_state):
    """
    Apply the rules to a column in the DataFrame
    """

    for rule in rules:
        if isinstance(rule, norma.rules.RuleProxy):
            rule = getattr(norma.engines.pyspark.rules, rule.name)(**rule.kwargs)  # noqa: PLW2901

        if getattr(rule, 'accumulates', False):
            df = rule.verify(df, column, error_state)
            continue

        df = error_state.flush(df)
        df = rule.verify(df, column, error_state)
        error_state.resync(df)

    return df


def _format_error_details(df, error_state) -> DataFrame:
    """
    Format the error details in the DataFrame
    """

    errors = {
        name: error_state.details_column_for(name)
        for name in error_state.names
        if error_state.details_column_for(name) in df.columns
    }

    empty = fn.array()
    entries = [
        fn.when(
            fn.array_size(fn.col(error)) > 0,
            fn.array(fn.struct(
                fn.lit(name).alias('key'),
                fn.struct(
                    fn.col(error).alias('details'),
                    _make_origin(df, name, error_state).alias('original'),
                ).alias('value'),
            )),
        ).otherwise(empty)
        for name, error in errors.items()
    ]

    details_map = fn.map_from_entries(fn.concat(*entries)) if entries else fn.create_map()
    details_map = details_map.cast(error_state.errors_dtype())

    if error_state.extras:
        details_map = fn.map_concat(details_map, *error_state.extras.values())

    return df.withColumn(error_state.error_column, details_map)


def _nullify_invalid_values(df, schema, error_state):
    """
    Reset invalid values to null after validation
    """

    error_state.resync(df)
    masks = {}
    for name in reversed(schema.nested_columns):
        if name in masks:
            # in reversed order an array column comes right after its element columns ("a[].b", "a[]", then "a"), so
            # its elements are reset here, before the checks of the array itself and of its parents: those checks then
            # wrap the reset array once, instead of being repeated inside every field of the parent struct
            _nullify_elements(name, masks.pop(name), error_state)
        if name not in error_state.names or not error_state.has_column(name):
            # nothing to reset: no error was reported for the column, or it was dropped along the way
            continue

        if '[]' not in name:
            details = error_state.details_column_for(name)
            if details in df.columns:
                error_state.set_expr(name, fn.when(fn.array_size(fn.col(details)) <= 0, error_state.expr_of(name)))
            continue

        indexes = error_state.indexes_column_for(name)
        if indexes in df.columns:
            array, _, path = name.partition('[]')
            masks.setdefault(array, []).append((path.removeprefix('.'), fn.col(indexes)))

    return error_state.flush(df)


def _nullify_elements(column, flags, error_state):
    """
    Reset the invalid values held by the elements of an array column
    """

    element_dtype = error_state.dtype_of(column).elementType
    zipped = fn.arrays_zip(
        error_state.expr_of(column).alias('_0'),
        *(flag.alias(f'_{i}') for i, (_, flag) in enumerate(flags, start=1)),
    )

    def element_of(item):
        value = item['_0']
        for i, (path, _) in enumerate(flags, start=1):
            flag = item[f'_{i}']
            if not path:
                value = fn.when(~flag, materialize(value, element_dtype))
                continue
            value = nested_set_expr(f'_.{path}', lambda x, flag=flag: fn.when(~flag, x), value, element_dtype)
        return materialize(value, element_dtype)

    error_state.set_expr(column, fn.transform(zipped, element_of))


def _make_origin(df, column, error_state):
    """
    Format the original value of a column for error reporting
    """

    backup = error_state.backup_column_for(column)
    if backup in df.columns:
        column = backup

    if '[]' in column:
        return format_value(nested_get_expr(column), 'array')
    return format_value(fn.col(column), dtype_at(df.schema, column).typeName())
