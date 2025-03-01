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


def make_test(value):
    @pytest.mark.parametrize('engine, case', [
        pytest.param(engine, prop, id=f'case #{i} | {engine}: {prop.get("description", "")}')
        for i, prop in enumerate(value)
        for engine in prop['engines']
    ])
    def func(spark_session, engine, case):
        {
            'pandas': make_test_pandas,
            'pyspark': partial(make_test_pyspark, spark_session)
        }[engine](case)

    return func


for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), 'data')):
    for file in files:
        if not file.endswith('.json'):
            continue

        test = file.split('.')[0]
        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
            cases = json.loads(f.read())
            globals()[f'test_{test}'] = make_test(cases)


def make_test_pandas(case):
    def json_serde(x):
        if pd.isna(x):
            return None
        return str(x)

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

        assert json.loads(json.dumps(actual.to_dict(orient='records'), default=json_serde)) == case['then']['data']
        return

    assert e.type.__name__ == case['then']['raises']['type']


def make_test_pyspark(spark_session, case):
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
