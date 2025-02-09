import pandas

from norma.engines.pandas import rules, validator


def test_schema_validate():
    schema = {
        'col1': [
            rules.int_parsing(),
            rules.greater_than(1),
            rules.multiple_of(2)
        ],
        'col2': [
            rules.str_parsing(),
            rules.pattern('^bar$')
        ]
    }
    df = pandas.DataFrame({
        'col1': [1, '2', 'unknown'],
        'col2': ['foo', 'bar', 'bar']
    })

    actual = validator.validate(schema, df, False)

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
