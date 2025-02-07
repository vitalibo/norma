import json
import os
from unittest import mock

import pytest
from pyxis.pyspark import StructType

from norma.pyspark import rules
from norma.pyspark.rules import ErrorState


def generate_test(value):
    @pytest.mark.parametrize('case', [
        pytest.param(prop, id=f'case #{i}: {prop.get("description", "")}')
        for i, prop in enumerate(value['cases'])
    ])
    def test_func(spark_session, case: dict):
        # given
        df = spark_session.createDataFrame(
            case['given']['data'], StructType.from_json(case['given']['schema']))
        error_state = ErrorState('errors')

        with (
            pytest.raises(Exception, match=case['then']['raises']['match'])
            if 'raises' in case['then'] else mock.MagicMock()
        ) as e:
            # when
            rule = getattr(rules, value['test'])(**case['when']['args'])
            actual = rule.verify(df, case['when'].get('column', 'col'), error_state)
            # then
            assert actual.schema.json() == StructType.from_json(case['then']['schema']).json()
            assert actual.toJSON().map(json.loads).collect() == case['then']['data']
            return
        assert e.type.__name__ == case['then']['raises']['type']

    return test_func


with open(os.path.join(os.path.dirname(__file__), 'data/rules.json'), 'r', encoding='utf-8') as f:
    tests = json.loads(f.read())
    for test in tests:
        globals()[f'test_{test["test"]}'] = generate_test(test)
