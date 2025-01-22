from __future__ import annotations

from typing import Any, Dict, List, Union

import pandas as pd

import norma.rules
from norma.rules import ErrorState, Rule

__all__ = ['Column', 'Schema']


class Column:
    """
    A class to represent a column in a DataFrame.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
            self,
            dtype: type | str,
            *,
            rules: Union[Rule, List[Rule], None] = None,
            nullable: bool = True,
            eq: Any = None,
            ne: Any = None,
            gt: Any = None,
            lt: Any = None,
            ge: Any = None,
            le: Any = None,
            multiple_of: Any = None,
            min_length: int = None,
            max_length: int = None,
            pattern: str = None
    ) -> None:
        dtype_rules = {
            'int': norma.rules.int_parsing,
            'integer': norma.rules.int_parsing,
            'float': norma.rules.float_parsing,
            'double': norma.rules.float_parsing,
            'string': norma.rules.string_parsing,
            'str': norma.rules.string_parsing,
            'boolean': norma.rules.boolean_parsing,
            'bool': norma.rules.boolean_parsing,
        }

        dtype = dtype.__name__ if isinstance(dtype, type) else dtype
        if dtype not in dtype_rules:
            raise ValueError(f"Unsupported dtype '{dtype}'")

        def apply_if(condition, rule):
            return (rule(condition),) if condition else ()

        self.rules = [
            *apply_if(not nullable, lambda x: norma.rules.required()),
            dtype_rules[dtype](),
            *apply_if(eq, norma.rules.equal_to),
            *apply_if(ne, norma.rules.not_equal_to),
            *apply_if(gt, norma.rules.greater_than),
            *apply_if(lt, norma.rules.less_than),
            *apply_if(ge, norma.rules.greater_than_equal),
            *apply_if(le, norma.rules.less_than_equal),
            *apply_if(multiple_of, norma.rules.multiple_of),
            *apply_if(min_length, norma.rules.min_length),
            *apply_if(max_length, norma.rules.max_length),
            *apply_if(pattern, norma.rules.pattern),
        ]

        if rules is not None:
            self.rules.extend([rules] if isinstance(rules, Rule) else rules)


class Schema:
    """
    A class to represent a schema for a DataFrame.
    """

    def __init__(self, columns: Dict[str, Column]) -> None:
        self.columns = columns

    def validate(self, df: pd.DataFrame, error_column: str = 'errors') -> pd.DataFrame:
        """
        Validate an input DataFrame according to the schema and introduce an error column that contains the errors.
        """

        original_df = df.copy()

        error_state = ErrorState(df.index)
        for column in self.columns:
            for rule in self.columns[column].rules:
                series = rule.verify(df, column=column, error_state=error_state)
                if series is not None:
                    df[column] = series

        for index in error_state.errors:  # pylint: disable=consider-using-dict-items
            for column in error_state.errors[index]:
                error_state.errors[index][column]['original'] = original_df.loc[index, column]

        for column in error_state.masks:
            df.loc[error_state.masks[column], column] = None

        df[error_column] = df.index.map(error_state.errors)
        return df[original_df.columns.tolist() + [error_column]]

    @staticmethod
    def from_json_schema(json_schema: Dict) -> Schema:
        pass
