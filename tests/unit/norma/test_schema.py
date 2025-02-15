import json
import os
from functools import partial
from unittest import mock

import pandas as pd
import pyspark.sql.functions as fn  # noqa pylint: disable=unused-import
import pytest
from pyxis.pyspark import StructType

from norma import rules  # noqa pylint: disable=unused-import
from norma.schema import Column, Schema


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


def generate_test(value):
    def crete_schema(o):
        return Schema(**{
            sk: {
                ck: Column(**{
                    # pylint: disable=eval-used
                    k: eval(v['expr'], globals()) if isinstance(v, dict) and 'expr' in v else v
                    for k, v in cv.items()
                })
                for ck, cv in sv.items()
            } if sk == 'columns' else sv
            for sk, sv in o.items()
        })

    @pytest.mark.parametrize('engine, case', [
        pytest.param(engine, prop, id=f'case #{i} | {engine}: {prop.get("description", "")}')
        for i, prop in enumerate(value)
        for engine in prop['engines']
    ])
    def test_func(spark_session, engine, case):
        {
            'pandas': test_func_pandas,
            'pyspark': partial(test_func_pyspark, spark_session)
        }[engine](case)

    def test_func_pandas(case):
        if 'schema' in case['given']:
            df = pd.DataFrame({
                k: pd.Series([v[k] for v in case['given']['data']], dtype=t)
                for k, t in case['given']['schema']['pandas'].items()
            })
        else:
            df = pd.DataFrame(case['given']['data'])
            df = df.convert_dtypes()

        with (
                pytest.raises(Exception, match=case['then']['raises']['match'])
                if 'raises' in case['then'] else mock.MagicMock()
        ) as e:
            schema = (
                Schema.from_json_schema(case['when']['json_schema'])
                if 'json_schema' in case['when'] else
                crete_schema(case['when']['schema'])
            )
            actual = schema.validate(df)

            assert actual.to_dict(orient='records') == case['then']['data']
            return

        assert e.type.__name__ == case['then']['raises']['type']

    def test_func_pyspark(spark_session, case):
        params = {}
        if 'schema' in case['given']:
            params['schema'] = StructType.from_json(case['given']['schema']['pyspark'])

        df = spark_session.createDataFrame(case['given']['data'], **params)

        with (
                pytest.raises(Exception, match=case['then']['raises']['match'])
                if 'raises' in case['then'] else mock.MagicMock()
        ) as e:
            schema = (
                Schema.from_json_schema(case['when']['json_schema'])
                if 'json_schema' in case['when'] else
                crete_schema(case['when']['schema'])
            )
            actual = schema.validate(df)

            assert list(map(json.loads, actual.toJSON().collect())) == case['then']['data']
            return

        assert e.type.__name__ == case['then']['raises']['type']

    return test_func


for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), 'data')):
    for file in files:
        if not file.endswith('.json'):
            continue

        test = file.split('.')[0]
        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
            cases = json.loads(f.read())
            globals()[f'test_{test}'] = generate_test(cases)
