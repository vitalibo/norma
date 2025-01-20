import unittest.mock

import numpy as np
import pandas
import pytest

import norma
from norma import rules
from norma.schema import Column


@pytest.mark.parametrize('kwargs, expected', [
    ({'dtype': int}, lambda mock_rules: [mock_rules.int_parsing()]),
    ({'dtype': 'int'}, lambda mock_rules: [mock_rules.int_parsing()]),
    ({'dtype': 'integer'}, lambda mock_rules: [mock_rules.int_parsing()]),
    ({'dtype': float}, lambda mock_rules: [mock_rules.float_parsing()]),
    ({'dtype': 'float'}, lambda mock_rules: [mock_rules.float_parsing()]),
    ({'dtype': 'double'}, lambda mock_rules: [mock_rules.float_parsing()]),
    ({'dtype': str}, lambda mock_rules: [mock_rules.string_parsing()]),
    ({'dtype': 'string'}, lambda mock_rules: [mock_rules.string_parsing()]),
    ({'dtype': 'str'}, lambda mock_rules: [mock_rules.string_parsing()]),
    ({'dtype': bool}, lambda mock_rules: [mock_rules.boolean_parsing()]),
    ({'dtype': 'boolean'}, lambda mock_rules: [mock_rules.boolean_parsing()]),
    ({'dtype': 'bool'}, lambda mock_rules: [mock_rules.boolean_parsing()]),
    ({'dtype': 'bool', 'nullable': False}, lambda mock_rules: [mock_rules.required(), mock_rules.boolean_parsing()]),
    ({'dtype': 'str', 'eq': 'foo'}, lambda mock_rules: [mock_rules.string_parsing(), mock_rules.equal_to('foo')]),
    ({'dtype': 'str', 'ne': 'foo'}, lambda mock_rules: [mock_rules.string_parsing(), mock_rules.not_equal_to('foo')]),
    ({'dtype': 'int', 'gt': 10}, lambda mock_rules: [mock_rules.int_parsing(), mock_rules.greater_than(10)]),
    ({'dtype': 'int', 'lt': 10}, lambda mock_rules: [mock_rules.int_parsing(), mock_rules.less_than(10)]),
    ({'dtype': 'int', 'ge': 10}, lambda mock_rules: [mock_rules.int_parsing(), mock_rules.greater_than_equal(10)]),
    ({'dtype': 'int', 'le': 10}, lambda mock_rules: [mock_rules.int_parsing(), mock_rules.less_than_equal(10)]),
    ({'dtype': 'int', 'multiple_of': 10}, lambda mock_rules: [mock_rules.int_parsing(), mock_rules.multiple_of(10)]),
    ({'dtype': 'str', 'min_length': 10}, lambda mock_rules: [mock_rules.string_parsing(), mock_rules.min_length(10)]),
    ({'dtype': 'str', 'max_length': 10}, lambda mock_rules: [mock_rules.string_parsing(), mock_rules.max_length(10)]),
    ({'dtype': 'str', 'pattern': 'foo'}, lambda mock_rules: [mock_rules.string_parsing(), mock_rules.pattern('foo')]),
])
def test_column(kwargs, expected):
    with unittest.mock.patch('norma.rules') as mock_rules:
        column = Column(**kwargs)

        assert column.rules == expected(mock_rules)


def test_column_rule():
    with unittest.mock.patch('norma.rules') as mock_rules:
        rule = rules.equal_to(10)
        column = Column(str, rules=rule)

        assert column.rules == [mock_rules.string_parsing(), rule]


def test_column_rules():
    with unittest.mock.patch('norma.rules') as mock_rules:
        rule1 = rules.equal_to(10)
        rule2 = rules.not_equal_to(22)
        column = Column(str, rules=[rule1, rule2])

        assert column.rules == [mock_rules.string_parsing(), rule1, rule2]


def test_schema_validate():
    schema = norma.schema.Schema({
        'col1': Column(int, rules=[
            rules.greater_than(1),
            rules.multiple_of(2)
        ]),
        'col2': Column(str, rules=rules.pattern('^bar$'))
    })
    df = pandas.DataFrame({
        'col1': [1, '2', 'unknown'],
        'col2': ['foo', 'bar', 'bar']
    })

    actual = schema.validate(df)

    assert actual.to_dict(orient='records') == [
        {
            'col1': None,
            'col2': None,
            'errors': {
                'col1': {
                    'details': [
                        {
                            'type': 'greater_than',
                            'msg': 'Input should be greater than 1'
                        },
                        {
                            'type': 'multiple_of',
                            'msg': 'Input should be a multiple of 2'
                        }
                    ],
                    'original': 1
                },
                'col2': {
                    'details': [
                        {
                            'type': 'pattern',
                            'msg': 'Input should match the pattern ^bar$'
                        }
                    ],
                    'original': 'foo'
                }
            }
        },
        {
            'col1': 2,
            'col2': 'bar',
            'errors': np.nan
        },
        {
            'col1': None,
            'col2': 'bar',
            'errors': {
                'col1': {
                    'details': [
                        {
                            'type': 'int_parsing',
                            'msg': 'Input should be a valid integer, unable to parse value as an integer'
                        }
                    ],
                    'original': 'unknown'
                }
            }
        }
    ]
