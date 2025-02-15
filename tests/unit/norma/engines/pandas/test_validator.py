import pandas

from norma.engines.pandas import rules, validator
from norma.schema import Column, Schema


def test_schema_validate():
    schema = Schema({
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

    actual = validator.validate(schema, df)

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
                    'original': '1'
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
                            'type': 'int_parsing',
                            'msg': 'Input should be a valid integer, unable to parse value as an integer'
                        }
                    ],
                    'original': '"unknown"'
                }
            }
        }
    ]
