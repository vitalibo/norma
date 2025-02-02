import json

import norma.pyspark.schema
from norma.pyspark import rules
from norma.pyspark.schema import Column


def test_schema_validate(spark):
    schema = norma.pyspark.schema.Schema({
        'col1': Column(int, rules=[
            rules.greater_than(1),
            rules.multiple_of(2)
        ]),
        'col2': Column(str, rules=rules.pattern('^bar$'))
    })

    df = spark.spark_session.createDataFrame([
        (1, 'foo'),
        ('2', 'bar'),
        ('unknown', 'bar')
    ], ['col1', 'col2'])

    actual = schema.validate(df)

    assert list(map(json.loads, actual.toJSON().collect())) == [
        {
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
                    'original': '1'
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
            'errors': {}
        },
        {
            'col2': 'bar',
            'errors': {
                'col1': {
                    'details': [
                        {
                            'msg': 'Input should be a valid integer, unable to parse value as an integer',
                            'type': 'int_parsing'
                        }
                    ],
                    'original': 'unknown'
                }
            }
        }
    ]
