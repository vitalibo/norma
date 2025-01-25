from unittest import mock

import pandas
import pytest

import norma
from norma import rules
from norma.schema import Column, Schema


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
    with mock.patch('norma.rules') as mock_rules:
        column = Column(**kwargs)

        assert column.rules == expected(mock_rules)


def test_column_rule():
    with mock.patch('norma.rules') as mock_rules:
        rule = rules.equal_to(10)
        column = Column(str, rules=rule)

        assert column.rules == [mock_rules.string_parsing(), rule]


def test_column_rules():
    with mock.patch('norma.rules') as mock_rules:
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
            'errors': None
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


def test_schema_validate_allow_extra():
    schema = norma.schema.Schema(
        {
            'col1': Column(str),
            'col2': Column(str)
        },
        allow_extra=True
    )
    df = pandas.DataFrame({
        'col1': ['foo', 'baz'],
        'col2': ['bar', 'qux'],
        'col3': ['qux', 123]
    })

    actual = schema.validate(df)

    assert actual.to_dict(orient='records') == [
        {
            'col1': 'foo',
            'col2': 'bar',
            'col3': 'qux',
            'errors': None
        },
        {
            'col1': 'baz',
            'col2': 'qux',
            'col3': 123,
            'errors': None
        }
    ]


def test_schema_validate_forbidden_extra():
    schema = norma.schema.Schema(
        {
            'col1': Column(str),
            'col2': Column(str)
        },
        allow_extra=False
    )
    df = pandas.DataFrame({
        'col1': ['foo', 'baz'],
        'col2': ['bar', 'qux'],
        'col3': ['qux', 123]
    })

    actual = schema.validate(df)

    assert actual.to_dict(orient='records') == [
        {
            'col1': 'foo',
            'col2': 'bar',
            'errors': {
                'col3': {
                    'details': [
                        {
                            'msg': 'Extra inputs are not permitted',
                            'type': 'extra_forbidden'
                        }
                    ],
                    'original': 'qux'
                }
            }
        },
        {
            'col1': 'baz',
            'col2': 'qux',
            'errors': {
                'col3': {
                    'details': [
                        {
                            'msg': 'Extra inputs are not permitted',
                            'type': 'extra_forbidden'
                        }
                    ],
                    'original': 123
                }
            }
        }
    ]


def test_schema_from_json_schema():
    json_schema = {
        '$id': 'https://norma.github.com/dataframe.schema.json',
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'DataFrame',
        'type': 'object',
        'properties': {
            'name': {
                'type': 'string',
                'description': 'The person\'s full name.',
                'minLength': 3,
                'maxLength': 256,
                'pattern': '^[A-Za-z ]+$'
            },
            'age': {
                'description': 'Age in years which must be equal to or greater than zero.',
                'type': 'integer',
                'minimum': 0,
                'maximum': 120
            },
            'height': {
                'description': 'Height in meters which must be greater than 0.5 and less than 3.0.',
                'type': 'number',
                'exclusiveMinimum': 0.5,
                'exclusiveMaximum': 3.0
            },
            'disabled': {
                'description': 'A boolean flag to indicate if the person is disabled.',
                'type': 'boolean'
            }
        },
        'required': ['name', 'age', 'disabled'],
        'additionalProperties': False
    }

    with mock.patch('norma.schema.Column') as mock_column:
        actual = Schema.from_json_schema(json_schema)

    assert actual.columns.keys() == {'name', 'age', 'height', 'disabled'}
    assert mock_column.mock_calls == [
        mock.call('string', nullable=False, min_length=3, max_length=256, pattern='^[A-Za-z ]+$'),
        mock.call('integer', nullable=False, ge=0, le=120),
        mock.call('number', nullable=True, gt=0.5, lt=3.0),
        mock.call('boolean', nullable=False)
    ]
    assert actual.allow_extra is False
