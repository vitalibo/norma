import json
import os
import uuid  # noqa: F401
from functools import partial
from unittest import mock

import pandas as pd
import pyspark.sql.functions as fn  # noqa: F401
import pytest
from pyxis.pyspark import StructType

from norma import rules  # noqa: F401
from norma.schema import Column, Schema

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEFINITIONS_DIR = os.path.join(DATA_DIR, 'definitions')
REF_PREFIX = '!ref #/definitions/'


def make_test(value):
    @pytest.mark.parametrize(
        ('engine', 'case'),
        [
            pytest.param(engine, variant, id=f'case #{i} | {name}: {prop.get("description", "")}')
            for i, prop in enumerate(value)
            for engine in prop['engines']
            for name, variant in schema_variants(engine, prop)
        ],
    )
    def func(spark_session, engine, case):
        {
            'pandas': make_test_pandas,
            'pyspark': partial(make_test_pyspark, spark_session),
        }[engine](case)

    return func


def schema_variants(engine, case):
    """
    Expand a case whose input schema of the engine is a list into one case per schema, sharing the data and the
    expectations
    """

    schemas = case['given'].get('schema', {}).get(engine)
    if not isinstance(schemas, list):
        yield engine, case
        return

    for j, schema in enumerate(schemas):
        given = {**case['given'], 'schema': {**case['given']['schema'], engine: schema}}
        yield f'{engine}[{j}]', {**case, 'given': given}


def resolve_refs(node):
    if isinstance(node, dict):
        return {key: resolve_refs(value) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve_refs(value) for value in node]
    if isinstance(node, str) and node.startswith(REF_PREFIX):
        with open(os.path.join(DEFINITIONS_DIR, f'{node.removeprefix(REF_PREFIX)}.json'), encoding='utf-8') as f:
            return resolve_refs(json.loads(f.read()))
    return node


for root, _, files in os.walk(DATA_DIR):
    if root == DEFINITIONS_DIR:
        continue

    for file in files:
        if not file.endswith('.json'):
            continue

        test = file.split('.')[0]
        with open(os.path.join(root, file), encoding='utf-8') as f:
            cases = resolve_refs(json.loads(f.read()))
            globals()[f'test_{test}'] = make_test(cases)


def make_test_pandas(case):
    def json_serde(x):
        if pd.isna(x):
            return None
        return str(x)

    def as_value(v):
        if isinstance(v, dict) and set(v) == {'expr'}:
            return eval(v['expr'], globals())  # noqa: S307
        return v

    if 'schema' in case['given'] and 'pandas' in case['given']['schema']:
        df = pd.DataFrame(
            {
                k: pd.Series([as_value(v[k]) for v in case['given']['data']], dtype=t)
                for k, t in case['given']['schema']['pandas'].items()
            }
        )
    else:
        df = pd.DataFrame(case['given']['data'])
        df = df.convert_dtypes()

    with (
        pytest.raises(Exception, match=case['then']['raises']['match'])
        if 'raises' in case['then']
        else mock.MagicMock()
    ) as e:
        schema = (
            Schema.from_json_schema(case['when']['json_schema'])
            if 'json_schema' in case['when']
            else crete_schema(case['when']['schema'])
        )
        actual = schema.validate(df, error_column=case['when'].get('error_column', 'errors'))

        assert json.loads(json.dumps(actual.to_dict(orient='records'), default=json_serde)) == case['then']['data']
        return

    assert e.type.__name__ == case['then']['raises']['type']


def make_test_pyspark(spark_session, case):
    df = spark_session.createDataFrame(
        case['given']['data'], schema=StructType.from_json(case['given']['schema']['pyspark'])
    )

    with (
        pytest.raises(Exception, match=case['then']['raises']['match'])
        if 'raises' in case['then']
        else mock.MagicMock()
    ) as e:
        schema = (
            Schema.from_json_schema(case['when']['json_schema'])
            if 'json_schema' in case['when']
            else crete_schema(case['when']['schema'])
        )
        actual = schema.validate(df, error_column=case['when'].get('error_column', 'errors'))

        assert list(map(json.loads, actual.toJSON().collect())) == case['then']['data']
        assert actual.schema.json() == StructType.from_json(case['then']['schema']['pyspark']).json()
        return

    assert e.type.__name__ == case['then']['raises']['type']


def crete_schema(o):
    return Schema(**{
        sk: {
            ck: Column(**{
                k: eval(v['expr'], globals()) if isinstance(v, dict) and 'expr' in v else v  # noqa: S307
                for k, v in cv.items()
            })
            for ck, cv in sv.items()
        }
        if sk == 'columns'
        else sv
        for sk, sv in o.items()
    })
