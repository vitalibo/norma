from __future__ import annotations

from typing import Dict, Union, List

import pandas as pd

from norma.rules import Rule, ErrorState


class Column:

    def __init__(
            self,
            dtype: type,
            *,
            rules: Union[Rule, List[Rule], None] = None,
            nullable: bool = True,
    ) -> None:
        self.rules = []
        if rules is not None:
            self.rules.extend(rules if isinstance(rules, list) else [rules])


class Schema:

    def __init__(self, schema: Dict[str, Column]) -> None:
        self.rules = [(name, rule) for name, column in schema.items() for rule in column.rules]

    def validate(
            self,
            df: pd.DataFrame,
            error_column: str = 'errors'
    ) -> pd.DataFrame:
        original_df = df.copy()

        error_state = ErrorState(df.index)
        for column, rule in self.rules:
            series = rule.verify(df, column=column, error_state=error_state)
            if series is not None:
                df[column] = series

        for index in error_state.errors:
            for column in error_state.errors[index]:
                error_state.errors[index][column]['original'] = original_df.loc[index, column]

        for column in error_state.masks:
            df.loc[error_state.masks[column], column] = None

        df[error_column] = df.index.map(error_state.errors)
        return df[original_df.columns.tolist() + [error_column]]

    @staticmethod
    def from_json_schema(json_schema: Dict) -> Schema:
        pass
