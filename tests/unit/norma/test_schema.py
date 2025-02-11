import json
from unittest import mock

import pandas
import pytest
from pyspark.sql import functions as fn

import norma
from norma import rules
from norma.schema import Column, Schema


@pytest.mark.parametrize('kwargs, expected', [
    ({'dtype': int}, lambda x: [x.int_parsing()]),
    ({'dtype': 'int'}, lambda x: [x.int_parsing()]),
    ({'dtype': 'integer'}, lambda x: [x.int_parsing()]),
    ({'dtype': float}, lambda x: [x.float_parsing()]),
    ({'dtype': 'float'}, lambda x: [x.float_parsing()]),
    ({'dtype': 'double'}, lambda x: [x.float_parsing()]),
    ({'dtype': str}, lambda x: [x.str_parsing()]),
    ({'dtype': 'string'}, lambda x: [x.str_parsing()]),
    ({'dtype': 'str'}, lambda x: [x.str_parsing()]),
    ({'dtype': bool}, lambda x: [x.bool_parsing()]),
    ({'dtype': 'boolean'}, lambda x: [x.bool_parsing()]),
    ({'dtype': 'bool'}, lambda x: [x.bool_parsing()]),
    # ({'dtype': 'bool', 'nullable': False}, lambda x: [x.required(), x.bool_parsing()]),
    # ({'dtype': 'str', 'eq': 'foo'}, lambda x: [x.str_parsing(), x.equal_to('foo')]),
    # ({'dtype': 'str', 'ne': 'foo'}, lambda x: [x.str_parsing(), x.not_equal_to('foo')]),
    # ({'dtype': 'int', 'gt': 10}, lambda x: [x.int_parsing(), x.greater_than(10)]),
    # ({'dtype': 'int', 'lt': 10}, lambda x: [x.int_parsing(), x.less_than(10)]),
    # ({'dtype': 'int', 'ge': 10}, lambda x: [x.int_parsing(), x.greater_than_equal(10)]),
    # ({'dtype': 'int', 'le': 10}, lambda x: [x.int_parsing(), x.less_than_equal(10)]),
    # ({'dtype': 'int', 'multiple_of': 10}, lambda x: [x.int_parsing(), x.multiple_of(10)]),
    # ({'dtype': 'str', 'min_length': 10}, lambda x: [x.str_parsing(), x.min_length(10)]),
    # ({'dtype': 'str', 'max_length': 10}, lambda x: [x.str_parsing(), x.max_length(10)]),
    # ({'dtype': 'str', 'pattern': 'foo'}, lambda x: [x.str_parsing(), x.pattern('foo')]),
    # ({'dtype': 'str', 'isin': ['foo', 'bar']}, lambda x: [x.str_parsing(), x.isin(['foo', 'bar'])]),
    # ({'dtype': 'str', 'notin': ['foo', 'bar']}, lambda x: [x.str_parsing(), x.notin(['foo', 'bar'])]),
    # ({'dtype': 'date'}, lambda x: [x.date_parsing()]),
    # ({'dtype': 'datetime'}, lambda x: [x.datetime_parsing()]),
    # ({'dtype': 'timestamp'}, lambda x: [x.timestamp_parsing()]),
    # ({'dtype': 'timestamp[s]'}, lambda x: [x.timestamp_parsing('s')]),
    # ({'dtype': 'timestamp[ms]'}, lambda x: [x.timestamp_parsing('ms')]),
    # ({'dtype': 'time'}, lambda x: [x.time_parsing()]),
])
def test_column(kwargs, expected):
    with mock.patch('norma.rules') as mock_rules:
        column = Column(**kwargs)

        assert column.rules == expected(mock_rules)


def test_column_rule():
    with mock.patch('norma.rules') as mock_rules:
        mock_rules.str_parsing.return_value.priority = 1

        rule = rules.equal_to(10)
        column = Column(str, rules=rule)

        assert column.rules == [mock_rules.str_parsing(), rule]


def test_column_rules():
    with mock.patch('norma.rules') as mock_rules:
        mock_rules.str_parsing.return_value.priority = 1

        rule1 = rules.equal_to(10)
        rule2 = rules.not_equal_to(22)
        column = Column(str, rules=[rule1, rule2])

        assert column.rules == [mock_rules.str_parsing(), rule1, rule2]


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
            'errors': {}
        },
        {
            'col1': 'baz',
            'col2': 'qux',
            'col3': 123,
            'errors': {}
        }
    ]


def test_schema_validate_allow_extra_pyspark(spark_session):
    schema = norma.schema.Schema(
        {
            'col1': Column(str),
            'col2': Column(str)
        },
        allow_extra=True
    )
    df = spark_session.createDataFrame([
        ('foo', 'bar', 'qux'),
        ('baz', 'qux', '123')
    ], ['col1', 'col2', 'col3'])

    actual = schema.validate(df)

    assert list(map(json.loads, actual.toJSON().collect())) == [
        {
            'col1': 'foo',
            'col2': 'bar',
            'col3': 'qux',
            'errors': {}
        },
        {
            'col1': 'baz',
            'col2': 'qux',
            'col3': '123',
            'errors': {}
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


def test_schema_validate_forbidden_extra_pyspark(spark_session):
    schema = norma.schema.Schema(
        {
            'col1': Column(str),
            'col2': Column(str)
        },
        allow_extra=False
    )
    df = spark_session.createDataFrame([
        ('foo', 'bar', 'qux'),
        ('baz', 'qux', '123')
    ], ['col1', 'col2', 'col3'])

    actual = schema.validate(df)

    assert list(map(json.loads, actual.toJSON().collect())) == [
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
                    'original': '123'
                }
            }
        }
    ]


def test_schema_validate_default_pandas():
    schema = norma.schema.Schema({
        'col1': Column(str, default_factory=lambda x: x['col2'].str.upper()),
        'col2': Column(str, pattern='bar|qux', default='<default>')
    })
    df = pandas.DataFrame({
        'col1': ['foo', None, 'baz', None],
        'col2': ['bar', 'qux', None, None]
    })

    actual = schema.validate(df)

    assert actual.to_dict(orient='records') == [
        {
            'col1': 'foo',
            'col2': 'bar',
            'errors': {}
        },
        {
            'col1': 'QUX',
            'col2': 'qux',
            'errors': {}
        },
        {
            'col1': 'baz',
            'col2': '<default>',
            'errors': {}
        },
        {
            'col1': '<DEFAULT>',
            'col2': '<default>',
            'errors': {}
        }
    ]


def test_schema_validate_default_pyspark(spark_session):
    schema = norma.schema.Schema({
        'col1': Column(str, default_factory=lambda x: fn.upper(fn.col('col2'))),
        'col2': Column(str, pattern='bar|qux', default='<default>')
    })
    df = spark_session.createDataFrame([
        ('foo', 'bar'),
        (None, 'qux'),
        ('baz', None),
        (None, None)
    ], ['col1', 'col2'])

    actual = schema.validate(df)

    assert list(map(json.loads, actual.toJSON().collect())) == [
        {
            'col1': 'foo',
            'col2': 'bar',
            'errors': {}
        },
        {
            'col1': 'QUX',
            'col2': 'qux',
            'errors': {}
        },
        {
            'col1': 'baz',
            'col2': '<default>',
            'errors': {}
        },
        {
            'col1': '<DEFAULT>',
            'col2': '<default>',
            'errors': {}
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
                'maximum': 120,
                'enum': [0, 1, 2, 3]
            },
            'height': {
                'description': 'Height in meters which must be greater than 0.5 and less than 3.0.',
                'type': 'number',
                'exclusiveMinimum': 0.5,
                'exclusiveMaximum': 3.0
            },
            'disabled': {
                'description': 'A boolean flag to indicate if the person is disabled.',
                'type': 'boolean',
                'default': False
            },
            'releaseDate': {
                'type': 'string',
                'format': 'date-time'
            }
        },
        'required': ['name', 'age', 'disabled'],
        'additionalProperties': False
    }

    with mock.patch('norma.schema.Column') as mock_column:
        actual = Schema.from_json_schema(json_schema)

    assert actual.columns.keys() == {'name', 'age', 'height', 'disabled', 'releaseDate'}
    assert mock_column.mock_calls == [
        mock.call('string', nullable=False, min_length=3, max_length=256, pattern='^[A-Za-z ]+$'),
        mock.call('integer', nullable=False, ge=0, le=120, isin=[0, 1, 2, 3]),
        mock.call('number', nullable=True, gt=0.5, lt=3.0),
        mock.call('boolean', nullable=False, default=False),
        mock.call('datetime', nullable=True)
    ]
    assert actual.allow_extra is False
