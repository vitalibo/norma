from unittest import mock

from norma.schema import Schema


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
