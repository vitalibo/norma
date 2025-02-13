import json
import os
from functools import partial
from unittest import mock

import pandas as pd
import pyspark.sql.functions as fn  # noqa pylint: disable=unused-import
import pytest

from norma import rules
from norma.schema import Column, Schema


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
        for i, prop in enumerate(value['cases'])
        for engine in prop['engines']
    ])
    def test_func(spark_session, engine, case):
        {
            'pandas': test_func_pandas,
            'pyspark': partial(test_func_pyspark, spark_session)
        }[engine](case)

    def test_func_pandas(case):
        df = pd.DataFrame(case['given']['data'])

        schema = (
            Schema.from_json_schema(case['when']['json_schema'])
            if 'json_schema' in case['when'] else
            crete_schema(case['when']['schema'])
        )
        actual = schema.validate(df)

        assert actual.to_dict(orient='records') == case['then']['data']

    def test_func_pyspark(spark_session, case):
        df = spark_session.createDataFrame(case['given']['data'])

        schema = (
            Schema.from_json_schema(case['when']['json_schema'])
            if 'json_schema' in case['when'] else
            crete_schema(case['when']['schema'])
        )
        actual = schema.validate(df)

        assert list(map(json.loads, actual.toJSON().collect())) == case['then']['data']

    return test_func


with open(os.path.join(os.path.dirname(__file__), 'data/cases.json'), 'r', encoding='utf-8') as f:
    tests = json.loads(f.read())
    for test in tests:
        globals()[f'test_{test["test"]}'] = generate_test(test)
