from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import norma.rules
from norma.engines.pandas.rules import ErrorState, extra_forbidden

if TYPE_CHECKING:
    from norma.schema import Schema


def validate(
        schema: Schema, df: pd.DataFrame, error_column: str
) -> pd.DataFrame:
    """
    Validate the Pandas DataFrame according to the schema

    :param schema: The schema to validate the DataFrame against
    :param df: The DataFrame to validate
    :param error_column: The name of the column to store error information
    """

    has_array = False
    for name in schema.nested_columns:
        if name.count('[]') > 1:
            raise NotImplementedError('nested arrays are not supported yet')
        has_array = has_array or '[]' in name

    index = df.index
    df = df.reset_index(drop=True)
    original_df = df.copy()

    error_state = ErrorState(df.index, has_array=has_array)
    df = _validate(df, schema, error_state)
    _backfill_originals(error_state, original_df)

    df[error_column] = df.index.map(error_state.errors)
    df[error_column] = df[error_column].replace(np.nan, None).apply(lambda x: {} if x is None else x)
    df.index = index

    out_cols = original_df.columns if schema.allow_extra else schema.columns.keys()
    return df[[*out_cols, error_column]]


def _validate(df, schema, error_state, parent=''):
    """
    Recursively validate the DataFrame according to the schema
    """

    columns = {column.removeprefix(parent) for column in df.columns} | set(schema.columns)
    for column in columns:
        full_column = f'{parent}{column}'
        if full_column not in df.columns:
            df[full_column] = np.nan

        rules = list(schema.columns[column].rules) if column in schema.columns else []
        if not schema.allow_extra:
            rules.append(extra_forbidden([f'{parent}{allowed}' for allowed in schema.columns]))

        error_state.set_backup(full_column, df[full_column])
        for rule in rules:
            if isinstance(rule, norma.rules.RuleProxy):
                rule = getattr(norma.engines.pandas.rules, rule.name)(**rule.kwargs)  # noqa: PLW2901

            series = rule.verify(df, column=full_column, error_state=error_state)
            if series is not None:
                df[full_column] = series

        if column not in schema.columns or schema.columns[column].inner_schema is None:
            continue
        inner_schema = schema.columns[column].inner_schema

        if full_column in df.columns:
            if schema.columns[column].dtype in {'array', 'list'}:
                df = _validate_array(df, full_column, inner_schema, error_state)
            else:
                df = _validate_object(df, full_column, inner_schema, error_state)

    return _finalize(df, schema, error_state, parent)


def _validate_object(df, column, inner_schema, error_state):
    """
    Validate an object column by recursively validating a child frame built from its fields
    """

    child = pd.DataFrame(
        [value if isinstance(value, dict) else {} for value in df[column]], index=df.index, dtype='object'
    )
    child.columns = [f'{column}.{name}' for name in child.columns]
    child = _validate(child, inner_schema, error_state, parent=f'{column}.')

    names = list(inner_schema.columns)
    if inner_schema.allow_extra:
        names += [
            name.removeprefix(f'{column}.')
            for name in child.columns
            if name.removeprefix(f'{column}.') not in inner_schema.columns
        ]

    # null objects materialize as a dict of null fields, mirroring how pyspark rebuilds structs;
    # rows with errors are nullified afterward by the mask
    df[column] = pd.Series(
        [
            {name: _to_native(value) for name, value in zip(names, row)}
            for row in zip(*[child[f'{column}.{name}'].tolist() for name in names])
        ],
        index=df.index, dtype='object',
    )
    return df


def _validate_array(df, column, inner_column, error_state):
    """
    Validate an array column by exploding its elements into a child frame indexed by (row, position)
    """

    full_column = f'{column}[]'

    originals = df[column].tolist()
    rows, positions, values = [], [], []
    for row, value in zip(df.index, originals):
        if not isinstance(value, list):
            continue
        for position, element in enumerate(value):
            rows.append(row)
            positions.append(position)
            values.append(element)

    regrouped = {}
    if rows:
        child = pd.DataFrame({
            full_column: pd.Series(values, index=pd.MultiIndex.from_arrays([rows, positions]), dtype='object')
        })
        child = _validate_array_elements(child, full_column, inner_column, error_state)
        for row, element in zip(child.index.get_level_values(0), child[full_column].tolist()):
            regrouped.setdefault(row, []).append(_to_native(element))

    df[column] = pd.Series(
        [
            regrouped.get(row, value if isinstance(value, list) else None)
            for row, value in zip(df.index, originals)
        ],
        index=df.index, dtype='object',
    )
    return df


def _validate_array_elements(child, full_column, inner_column, error_state):
    """
    Run element-level rules over the exploded array elements and nullify the failing ones
    """

    error_state.set_backup(full_column, child[full_column])
    for rule in inner_column.rules:
        if isinstance(rule, norma.rules.RuleProxy):
            rule = getattr(norma.engines.pandas.rules, rule.name)(**rule.kwargs)  # noqa: PLW2901

        series = rule.verify(child, column=full_column, error_state=error_state)
        if series is not None:
            child[full_column] = series

    if full_column in error_state.masks:
        mask = error_state.masks[full_column].reindex(child.index, fill_value=False)
        if mask.any():
            child.loc[mask, full_column] = None

    if inner_column.default is not None:
        child[full_column] = child[full_column].fillna(inner_column.default)

    if inner_column.inner_schema is not None and inner_column.dtype == 'object':
        child = _validate_object(child, full_column, inner_column.inner_schema, error_state)

    return child


def _finalize(df, schema, error_state, parent=''):
    """
    Nullify invalid values and fill defaults
    """

    for full_column in df.columns:
        if full_column not in error_state.masks:
            continue
        mask = error_state.masks[full_column].reindex(df.index, fill_value=False)
        if not mask.any():
            continue
        df.loc[mask, full_column] = None

    for column in schema.columns:
        full_column = f'{parent}{column}'
        if full_column in df.columns and schema.columns[column].default is not None:
            df[full_column] = df[full_column].fillna(schema.columns[column].default)

    for column in schema.columns:
        full_column = f'{parent}{column}'
        if full_column in df.columns and schema.columns[column].default_factory is not None:
            df[full_column] = df[full_column].fillna(schema.columns[column].default_factory(df))

    return df


def _backfill_originals(error_state, original_df):
    """
    Render the original value for every recorded error
    """

    def render(index, column):
        if column in original_df.columns:
            return json.dumps(original_df.loc[index, column], separators=(',', ':'), default=_json_serde)

        if column not in error_state.backups:
            return 'null'

        backup = error_state.backups[column]
        if isinstance(backup.index, pd.MultiIndex):
            value = [_to_native(element) for element in backup.loc[index]] \
                if index in backup.index.get_level_values(0) else None
        else:
            value = _to_native(backup.loc[index]) if index in backup.index else None
        return json.dumps(value, separators=(',', ':'), default=_json_serde)

    for index in error_state.errors:
        for column in error_state.errors[index]:
            error_state.errors[index][column]['original'] = render(index, column)


def _to_native(value):
    """
    Convert pandas/numpy scalars to native Python values for JSON-compatible output
    """

    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _json_serde(obj):
    """
    Serialize an object to JSON. Used to serialize the original value
    """

    if pd.isna(obj):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)
