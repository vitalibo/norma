import numpy as np
import pandas as pd

from norma.models import ErrorState


class Validator:

    def __init__(self, rules):
        self.rules = rules

    def validate(self, df: pd.DataFrame) -> pd.Series:
        error_state = ErrorState(df.index)
        original_df = df.copy()

        for col, rules in self.rules.items():
            for rule in rules:
                new_df = rule(df, error_state)
                if new_df is not None:
                    df[col] = new_df

        for index in error_state.errors:
            for col in error_state.errors[index]:
                error_state.errors[index][col]['input'] = original_df.loc[index, col]

        for key, value in error_state.masks.items():
            df.loc[value, key] = np.nan

        return df.index.map(error_state.errors)
