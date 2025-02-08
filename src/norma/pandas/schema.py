from __future__ import annotations

from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd

import norma.pandas.rules
from norma.pandas.rules import ErrorState, Rule

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
            pattern: str = None,
            isin: List[Any] = None,
            notin: List[Any] = None,
            default: Any = None,
            default_factory: Callable[[pd.DataFrame], Any] = None,
    ) -> None:
        dtype_rules = {
            'int': norma.pandas.rules.int_parsing,
            'integer': norma.pandas.rules.int_parsing,
            'float': norma.pandas.rules.float_parsing,
            'double': norma.pandas.rules.float_parsing,
            'number': norma.pandas.rules.float_parsing,
            'string': norma.pandas.rules.string_parsing,
            'str': norma.pandas.rules.string_parsing,
            'boolean': norma.pandas.rules.boolean_parsing,
            'bool': norma.pandas.rules.boolean_parsing,
            'date': norma.pandas.rules.date_parsing,
            'datetime': norma.pandas.rules.datetime_parsing,
            'timestamp': norma.pandas.rules.timestamp_parsing,
            'timestamp[s]': lambda: norma.pandas.rules.timestamp_parsing('s'),
            'timestamp[ms]': lambda: norma.pandas.rules.timestamp_parsing('ms'),
            'time': norma.pandas.rules.time_parsing,
        }

        dtype = dtype.__name__ if isinstance(dtype, type) else dtype
        if dtype not in dtype_rules:
            raise ValueError(f"Unsupported dtype '{dtype}'")

        def apply_if(condition, rule):
            return (rule(condition),) if condition else ()

        self.rules = [
            *apply_if(not nullable, lambda x: norma.pandas.rules.required()),
            dtype_rules[dtype](),
            *apply_if(eq, norma.pandas.rules.equal_to),
            *apply_if(ne, norma.pandas.rules.not_equal_to),
            *apply_if(gt, norma.pandas.rules.greater_than),
            *apply_if(lt, norma.pandas.rules.less_than),
            *apply_if(ge, norma.pandas.rules.greater_than_equal),
            *apply_if(le, norma.pandas.rules.less_than_equal),
            *apply_if(multiple_of, norma.pandas.rules.multiple_of),
            *apply_if(min_length, norma.pandas.rules.min_length),
            *apply_if(max_length, norma.pandas.rules.max_length),
            *apply_if(pattern, norma.pandas.rules.pattern),
            *apply_if(isin, norma.pandas.rules.isin),
            *apply_if(notin, norma.pandas.rules.notin),
        ]

        if rules is not None:
            self.rules.extend([rules] if isinstance(rules, Rule) else rules)

        if default is not None and default_factory is not None:
            raise ValueError("Cannot specify both 'default' and 'default_factory'")

        self.default = default
        self.default_factory = default_factory


class Schema:
    """
    A class to represent a schema for a DataFrame.
    """

    def __init__(
            self,
            columns: Dict[str, Column],
            allow_extra: bool = False
    ) -> None:
        self.columns = columns
        self.allow_extra = allow_extra

    def validate(self, df: pd.DataFrame, error_column: str = 'errors') -> pd.DataFrame:
        """
        Validate an input DataFrame according to the schema and introduce an error column that contains the errors.
        """

        original_df = df.copy()

        error_state = ErrorState(df.index)
        for column in original_df.columns:
            rules = self.columns[column].rules if column in self.columns else []
            if not self.allow_extra:
                rules.append(
                    norma.pandas.rules.extra_forbidden(self.columns.keys()))

            for rule in rules:
                series = rule.verify(df, column=column, error_state=error_state)
                if series is not None:
                    df[column] = series

        for index in error_state.errors:  # pylint: disable=consider-using-dict-items
            for column in error_state.errors[index]:
                error_state.errors[index][column]['original'] = original_df.loc[index, column]

        for column in error_state.masks:
            df.loc[error_state.masks[column], column] = None

        for column in self.columns:
            if self.columns[column].default is not None:
                df[column] = df[column].fillna(self.columns[column].default)

        for column in self.columns:
            if self.columns[column].default_factory is not None:
                df[column] = df[column].fillna(self.columns[column].default_factory(df))

        df[error_column] = df.index.map(error_state.errors)
        df[error_column] = df[error_column].replace(np.nan, None)

        out_cols = original_df.columns if self.allow_extra else self.columns.keys()
        return df[list(out_cols) + [error_column]]

    @staticmethod
    def from_json_schema(json_schema: Dict) -> Schema:
        known = {
            'minimum': 'ge',
            'maximum': 'le',
            'exclusiveMinimum': 'gt',
            'exclusiveMaximum': 'lt',
            'multipleOf': 'multiple_of',
            'minLength': 'min_length',
            'maxLength': 'max_length',
            'pattern': 'pattern',
            'default': 'default',
            'enum': 'isin',
        }

        complex_types = {
            ('string', 'date'): 'date',
            ('string', 'date-time'): 'datetime',
            ('string', 'time'): 'time',
        }

        return Schema(
            {
                field: Column(
                    complex_types.get(
                        (properties.get('type'), properties.get('format')),
                        properties.get('type')
                    ),

                    nullable=field not in json_schema.get('required', []),
                    **{
                        known[key]: value
                        for key, value in properties.items()
                        if key in known
                    }
                )
                for field, properties in json_schema['properties'].items()
            },
            allow_extra=json_schema.get('additionalProperties', False)
        )
