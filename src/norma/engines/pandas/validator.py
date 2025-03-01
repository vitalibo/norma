import json

import numpy as np
import pandas as pd

import norma.rules
from norma.engines.pandas.rules import ErrorState, extra_forbidden


def validate(  # pylint: disable=too-many-branches
        schema: 'Schema', df: pd.DataFrame, error_column: str
) -> pd.DataFrame:
    """
    Validate the Pandas DataFrame according to the schema

    :param schema: The schema to validate the DataFrame against
    :param df: The DataFrame to validate
    :param error_column: The name of the column to store error information
    """

    original_df = df.copy()

    error_state = ErrorState(df.index)
    for column in set(list(original_df.columns) + list(schema.columns.keys())):
        rules = schema.columns[column].rules if column in schema.columns else []
        if not schema.allow_extra:
            rules.append(extra_forbidden(schema.columns.keys()))

        for rule in rules:
            if isinstance(rule, norma.rules.RuleProxy):
                rule = getattr(norma.engines.pandas.rules, rule.name)(**rule.kwargs)

            series = rule.verify(df, column=column, error_state=error_state)
            if series is not None:
                df[column] = series

    for index in error_state.errors:  # pylint: disable=consider-using-dict-items
        for column in error_state.errors[index]:
            if column in original_df.columns:
                error_state.errors[index][column]['original'] = \
                    json.dumps(original_df.loc[index, column], separators=(',', ':'), default=_json_serde)
            else:
                error_state.errors[index][column]['original'] = 'null'

    for column in error_state.masks:
        df.loc[error_state.masks[column], column] = None

    for column in schema.columns:
        if schema.columns[column].default is not None:
            df[column] = df[column].fillna(schema.columns[column].default)

    for column in schema.columns:
        if schema.columns[column].default_factory is not None:
            df[column] = df[column].fillna(schema.columns[column].default_factory(df))

    df[error_column] = df.index.map(error_state.errors)
    df[error_column] = df[error_column].replace(np.nan, None).apply(lambda x: {} if x is None else x)

    out_cols = original_df.columns if schema.allow_extra else schema.columns.keys()
    return df[list(out_cols) + [error_column]]


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
