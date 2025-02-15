from __future__ import annotations

from typing import Any, Callable, Dict, List, TypeVar, Union

import norma.rules
from norma.rules import Rule

try:
    from pandas import DataFrame as PandasDataFrame
except ImportError:
    PandasDataFrame = None

try:
    from pyspark.sql import DataFrame as PySparkDataFrame
except ImportError:
    PySparkDataFrame = None

T = TypeVar('T')


class Column:
    """
    Column definition for data validation
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
            self,
            dtype: Union[type, str],
            *,
            rules: Union[norma.rules.Rule, List[norma.rules.Rule], None] = None,
            nullable: bool = True,
            eq: Any = None,
            ne: Any = None,
            gt: Any = None,
            lt: Any = None,
            ge: Any = None,
            le: Any = None,
            multiple_of: Union[int, float, None] = None,
            min_length: Union[int, None] = None,
            max_length: Union[int, None] = None,
            pattern: Union[str, None] = None,
            isin: Union[List[Any], None] = None,
            notin: Union[List[Any], None] = None,
            default: Any = None,
            default_factory: Union[Callable, None] = None,
    ) -> None:
        dtype_parsing = {
            alias: parsing for parsing, aliases in {
                norma.rules.int_parsing: ['int', 'integer'],
                norma.rules.float_parsing: ['float', 'double', 'number'],
                norma.rules.str_parsing: ['string', 'str'],
                norma.rules.bool_parsing: ['boolean', 'bool'],
                norma.rules.datetime_parsing: ['datetime'],
                norma.rules.date_parsing: ['date'],
            }.items() for alias in aliases
        }

        dtype = dtype.__name__ if isinstance(dtype, type) else dtype
        if dtype not in dtype_parsing:
            raise ValueError(f"unsupported dtype '{dtype}'")

        if default is not None and default_factory is not None:
            raise ValueError('default and default_factory cannot be used together')

        defined_rules = {
            rule.name: rule for rule in [
                rule_definition(value) for rule_definition, value in [
                    (lambda _: norma.rules.required(), True if not nullable else None),
                    (lambda _: dtype_parsing[dtype](), True),
                    (norma.rules.equal_to, eq),
                    (norma.rules.not_equal_to, ne),
                    (norma.rules.greater_than, gt),
                    (norma.rules.greater_than_equal, ge),
                    (norma.rules.less_than, lt),
                    (norma.rules.less_than_equal, le),
                    (norma.rules.multiple_of, multiple_of),
                    (norma.rules.min_length, min_length),
                    (norma.rules.max_length, max_length),
                    (norma.rules.pattern, pattern),
                    (norma.rules.isin, isin),
                    (norma.rules.notin, notin),
                ] if value is not None
            ]
        }

        rules = [rules] if isinstance(rules, Rule) else rules
        for priority, rule in enumerate(rules or []):
            rule.priority = 6 + (priority + 1) / 10
            if 'name' in rule.__dict__:
                if rule.name in defined_rules:
                    raise ValueError(f"rule '{rule.name}' is already defined")
                defined_rules[rule.name] = rule
            else:
                defined_rules[str(rule)] = rule

        self.rules = sorted(defined_rules.values(), key=lambda x: x.priority)
        self.default = default
        self.default_factory = default_factory


class Schema:
    """
    Schema definition for data validation
    """

    def __init__(
            self,
            columns: Dict[str, Column],
            allow_extra: bool = False
    ) -> None:
        self.columns = columns
        self.allow_extra = allow_extra

    def validate(self, df: T, error_col: str = 'errors') -> T:
        # pylint: disable=import-outside-toplevel
        if PandasDataFrame and isinstance(df, PandasDataFrame):
            from norma.engines.pandas import validator
            return validator.validate(self, df, error_col)

        elif PySparkDataFrame and isinstance(df, PySparkDataFrame):
            from norma.engines.pyspark import validator
            return validator.validate(self, df, error_col)

        else:
            raise NotImplementedError('unsupported engine')

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
            'const': 'eq',
        }
        known_not = {
            'const': 'ne',
            'enum': 'notin',
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
                    },
                    **{
                        known_not[key]: value
                        for key, value in properties.get('not', {}).items()
                        if key in known_not
                    }
                )
                for field, properties in json_schema['properties'].items()
            },
            allow_extra=json_schema.get('additionalProperties', False)
        )
