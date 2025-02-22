import json

from norma.engines.pyspark import rules, validator
from norma.schema import Column, Schema


def test_schema_validate(spark_session):
    schema = Schema({
        'col1': Column(int, rules=[
            rules.greater_than(1),
            rules.multiple_of(2)
        ]),
        'col2': Column(str, rules=rules.pattern('^bar$'))
    })

    df = spark_session.createDataFrame([
        (1, 'foo'),
        ('2', 'bar'),
        ('unknown', 'bar')
    ], ['col1', 'col2'])

    actual = validator.validate(schema, df)

    assert list(map(json.loads, actual.toJSON().collect())) == [
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
                    'original': '"1"'
                },
                'col2': {
                    'details': [
                        {
                            'type': 'string_pattern_mismatch',
                            'msg': 'String should match pattern "^bar$"'
                        }
                    ],
                    'original': '"foo"'
                }
            }
        },
        {
            'col1': 2,
            'col2': 'bar',
            'errors': {}
        },
        {
            'col1': None,
            'col2': 'bar',
            'errors': {
                'col1': {
                    'details': [
                        {
                            'msg': 'Input should be a valid integer, unable to parse string as an integer',
                            'type': 'int_parsing'
                        }
                    ],
                    'original': '"unknown"'
                }
            }
        }
    ]
