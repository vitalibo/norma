import json
import os
import uuid  # noqa pylint: disable=unused-import
from functools import partial
from unittest import mock

import numpy as np  # noqa pylint: disable=unused-import
import pandas as pd
import pyspark.sql.functions as fn  # noqa pylint: disable=unused-import
import pytest
from pyxis.pyspark import StructType

from norma import rules  # noqa pylint: disable=unused-import
from norma.engines.pandas import rules as pandas_rules
from norma.engines.pyspark import rules as pyspark_rules
from norma.schema import Column, Schema


def make_test(test_name, value):
    @pytest.mark.parametrize('engine, case', [
        pytest.param(engine, prop, id=f'case #{i} | {engine}: {prop.get("description", "")}')
        for i, prop in enumerate(value)
        for engine in prop['engines']
    ])
    def func(spark_session, engine, case):
        {
            'pandas': make_test_pandas,
            'pandas_api': partial(make_test_pandas_api, test_name),
            'pyspark': partial(make_test_pyspark, spark_session),
            'pyspark_api': partial(make_test_pyspark_api, spark_session, test_name),
        }[engine](case)

    return func


for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), 'data')):
    for file in files:
        if not file.endswith('.json'):
            continue

        test = file.split('.')[0]
        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
            cases = json.loads(f.read())
            globals()[f'test_{test}'] = make_test(test, cases)


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


def make_test_pyspark_api(spark_session, test_name, case):
    # given
    df = spark_session.createDataFrame(
        case['given']['data'], StructType.from_json(case['given']['schema']))
    error_state = pyspark_rules.ErrorState('errors', False)

    with (
            pytest.raises(Exception, match=case['then']['raises']['match'])
            if 'raises' in case['then'] else mock.MagicMock()
    ) as e:
        # when
        args = {
            k: eval(v['expr'], globals()) if isinstance(v, dict) and 'expr' in v else v  # pylint: disable=eval-used
            for k, v in case['when']['args'].items()
        }
        rule = getattr(pyspark_rules, test_name)(**args)
        column = case['when'].get('column', 'col')
        error_state.suffixes = {column: column}
        actual = rule.verify(df, column, error_state)

        # then
        assert actual.schema.json() == StructType.from_json(case['then']['schema']).json()
        assert actual.toJSON().map(json.loads).collect() == case['then']['data']
        return

    assert e.type.__name__ == case['then']['raises']['type']


def make_test_pandas_api(test_name, case):
    def as_data(o):
        if isinstance(o, dict):
            return {k: as_data(v) for k, v in o.items()}
        if isinstance(o, list):
            return [as_data(v) for v in o]
        if isinstance(o, str) and (o.startswith('pd.') or o.startswith('np.') or o.startswith('uuid.UUID(')):
            return eval(o)  # pylint: disable=eval-used
        return o

    def as_dtype(o):
        if o == "<class 'str'>":
            return str
        return o

    # given
    df = pd.DataFrame(as_data(case['given']['data']), dtype=as_dtype(case['given']['dtype']))
    error_state = pandas_rules.ErrorState(df.index)

    with (
            pytest.raises(Exception, match=case['then']['raises']['match'])
            if 'raises' in case['then'] else mock.MagicMock()
    ) as e:
        # when
        rule = getattr(pandas_rules, test_name)(**case['when']['args'])
        actual = rule.verify(df, case['when'].get('column', 'col'), error_state)

        # then
        if case['then']['data'] is not None:
            expected = pd.Series(as_data(case['then']['data']), dtype=as_dtype(case['then']['dtype']))
            assert actual.equals(expected)
        else:
            assert actual is None
        assert error_state.errors == {int(k): v for k, v in case['then']['errors'].items()}
        for key in set(error_state.masks.keys()).union(set(case['then']['masks'].keys())):
            try:
                assert error_state.masks[key].equals(pd.Series(case['then']['masks'][key], dtype=bool))
            except Exception as e:
                raise AssertionError(f'Error in key: {key}') from e
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
