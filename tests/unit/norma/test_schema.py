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
            },
            'address': {
                'description': 'Address of the person',
                'type': 'object',
                'properties': {
                    'street': {
                        'description': 'Street address',
                        'type': 'string'
                    },
                    'city': {
                        'description': 'City name',
                        'type': 'string'
                    },
                    'state': {
                        'description': 'State name',
                        'type': 'string'
                    },
                    'zip_code': {
                        'description': 'ZIP code',
                        'type': 'string'
                    }
                },
            }
        },
        'required': ['name', 'age', 'disabled'],
        'additionalProperties': False
    }

    with mock.patch('norma.schema.Column') as mock_column:
        actual = Schema.from_json_schema(json_schema)

    assert actual.columns.keys() == {'name', 'age', 'height', 'disabled', 'releaseDate', 'address'}
    assert mock_column.mock_calls == [
        mock.call('string', nullable=False, inner_schema=None, min_length=3, max_length=256, pattern='^[A-Za-z ]+$'),
        mock.call('integer', nullable=False, inner_schema=None, ge=0, le=120, isin=[0, 1, 2, 3]),
        mock.call('number', nullable=True, inner_schema=None, gt=0.5, lt=3.0),
        mock.call('boolean', nullable=False, inner_schema=None, default=False),
        mock.call('datetime', nullable=True, inner_schema=None),
        mock.call('string', nullable=True, inner_schema=None),
        mock.call('string', nullable=True, inner_schema=None),
        mock.call('string', nullable=True, inner_schema=None),
        mock.call('string', nullable=True, inner_schema=None),
        mock.call('object', nullable=True, inner_schema=mock.ANY)
    ]
    assert actual.allow_extra is False
