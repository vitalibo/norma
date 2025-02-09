from typing import Dict, List

import numpy as np
import pandas as pd

from norma.engines.pandas.rules import ErrorState, extra_forbidden
from norma.rules import Rule


def validate(
        columns: Dict[str, List[Rule]], df: pd.DataFrame, allow_extra: bool, error_column: str = 'errors'
) -> pd.DataFrame:
    original_df = df.copy()

    error_state = ErrorState(df.index)
    for column in original_df.columns:
        rules = columns[column] if column in columns else []
        if not allow_extra:
            rules.append(extra_forbidden(columns.keys()))

        for rule in rules:
            series = rule.verify(df, column=column, error_state=error_state)
            if series is not None:
                df[column] = series

    for index in error_state.errors:  # pylint: disable=consider-using-dict-items
        for column in error_state.errors[index]:
            error_state.errors[index][column]['original'] = original_df.loc[index, column]

    for column in error_state.masks:
        df.loc[error_state.masks[column], column] = None

    # for column in columns:
    #     if columns[column].default is not None:
    #         df[column] = df[column].fillna(columns[column].default)
    #
    # for column in columns:
    #     if columns[column].default_factory is not None:
    #         df[column] = df[column].fillna(columns[column].default_factory(df))

    df[error_column] = df.index.map(error_state.errors)
    df[error_column] = df[error_column].replace(np.nan, None)

    out_cols = original_df.columns if allow_extra else columns.keys()
    return df[list(out_cols) + [error_column]]
