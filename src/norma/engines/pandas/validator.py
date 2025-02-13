import numpy as np
import pandas as pd

import norma.rules
from norma.engines.pandas.rules import ErrorState, extra_forbidden


def validate(
        schema: 'Schema', df: pd.DataFrame, error_column: str = 'errors'
) -> pd.DataFrame:
    original_df = df.copy()

    error_state = ErrorState(df.index)
    for column in original_df.columns:
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
            error_state.errors[index][column]['original'] = original_df.loc[index, column]

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
