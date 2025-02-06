import json

import pytest
from pyspark.sql import functions as fn
from pyxis.pyspark import StructType

from norma.pyspark import rules
from norma.pyspark.rules import ErrorState


@pytest.mark.parametrize('case', [
    dict(in_col='col', in_dtype='string', in_value='foo',
         out_value={'col': 'foo', 'errors_col': [None]}),
    dict(in_col='col', in_dtype='string', in_value=None,
         out_value={'col': None, 'errors_col': [{'type': 'missing', 'msg': 'Field required'}]}),
    dict(in_col='col2', in_dtype='string', in_value=None, error_col='errors_col2',
         out_value={'col': None, 'col2': None, 'errors_col2': [{'type': 'missing', 'msg': 'Field required'}]}),
])
def test_required(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype'], case.get('error_col')))
    error_state = ErrorState('errors')
    rule = rules.required()

    actual = rule.verify(df, case['in_col'], error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [*[
    dict(**case, out_value={'col': case['in_value'], 'errors_col': [None]})
    for case in [
        dict(in_dtype='string', in_value='foo', equal_to='foo'),
        dict(in_dtype='byte', in_value=1, equal_to=1),
        dict(in_dtype='integer', in_value=1, equal_to=1),
        dict(in_dtype='float', in_value=1.0, equal_to=1.0),
        dict(in_dtype='boolean', in_value=True, equal_to=True),
    ]
], *[
    dict(**case, out_value={'col': case['in_value'],
                            'errors_col': [{'type': 'equal_to', 'msg': case['error_msg']}]})
    for case in [
        dict(in_dtype='string', in_value='foo', equal_to='bar', error_msg='Input should be equal to bar'),
        dict(in_dtype='integer', in_value=1, equal_to=2, error_msg='Input should be equal to 2'),
        dict(in_dtype='float', in_value=1.0, equal_to=2.0, error_msg='Input should be equal to 2.0'),
        dict(in_dtype='boolean', in_value=True, equal_to=False, error_msg='Input should be equal to False'),
    ]
]])
def test_equal_to(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.equal_to(case['equal_to'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [*[
    dict(**case, out_value={'col': case['in_value'], 'errors_col': [None]})
    for case in [
        dict(in_dtype='string', in_value='foo', not_equal_to='bar'),
        dict(in_dtype='byte', in_value=1, not_equal_to=2),
        dict(in_dtype='integer', in_value=1, not_equal_to=2),
        dict(in_dtype='float', in_value=1.0, not_equal_to=2.0),
        dict(in_dtype='boolean', in_value=True, not_equal_to=False),
    ]
], *[
    dict(**case, out_value={'col': case['in_value'],
                            'errors_col': [{'type': 'not_equal_to', 'msg': case['error_msg']}]})
    for case in [
        dict(in_dtype='string', in_value='foo', not_equal_to='foo', error_msg='Input should not be equal to foo'),
        dict(in_dtype='integer', in_value=1, not_equal_to=1, error_msg='Input should not be equal to 1'),
        dict(in_dtype='float', in_value=1.0, not_equal_to=1.0, error_msg='Input should not be equal to 1.0'),
        dict(in_dtype='boolean', in_value=True, not_equal_to=True, error_msg='Input should not be equal to True'),
    ]
]])
def test_not_equal_to(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.not_equal_to(case['not_equal_to'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [*[
    dict(**case, out_value={'col': case['in_value'], 'errors_col': [None]})
    for case in [
        dict(in_dtype='integer', in_value=1, greater_than=0),
        dict(in_dtype='float', in_value=1.0, greater_than=0.0),
    ]
], *[
    dict(**case, out_value={'col': case['in_value'],
                            'errors_col': [{'type': 'greater_than', 'msg': case['error_msg']}]})
    for case in [
        dict(in_dtype='integer', in_value=1, greater_than=1, error_msg='Input should be greater than 1'),
        dict(in_dtype='float', in_value=1.0, greater_than=1.0, error_msg='Input should be greater than 1.0'),
    ]
]])
def test_greater_than(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.greater_than(case['greater_than'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [*[
    dict(**case, out_value={'col': case['in_value'], 'errors_col': [None]})
    for case in [
        dict(in_dtype='integer', in_value=1, greater_than_equal=1),
        dict(in_dtype='float', in_value=1.0, greater_than_equal=1.0),
    ]
], *[
    dict(**case, out_value={'col': case['in_value'],
                            'errors_col': [{'type': 'greater_than_equal', 'msg': case['error_msg']}]})
    for case in [
        dict(in_dtype='integer', in_value=1, greater_than_equal=2,
             error_msg='Input should be greater than or equal to 2'),
        dict(in_dtype='float', in_value=1.0, greater_than_equal=2.0,
             error_msg='Input should be greater than or equal to 2.0'),
    ]
]])
def test_greater_than_equal(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.greater_than_equal(case['greater_than_equal'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [*[
    dict(**case, out_value={'col': case['in_value'], 'errors_col': [None]})
    for case in [
        dict(in_dtype='integer', in_value=1, less_than=2),
        dict(in_dtype='float', in_value=1.0, less_than=2.0),
    ]
], *[
    dict(**case, out_value={'col': case['in_value'],
                            'errors_col': [{'type': 'less_than', 'msg': case['error_msg']}]})
    for case in [
        dict(in_dtype='integer', in_value=1, less_than=1, error_msg='Input should be less than 1'),
        dict(in_dtype='float', in_value=1.0, less_than=1.0, error_msg='Input should be less than 1.0'),
    ]
]])
def test_less_than(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.less_than(case['less_than'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [*[
    dict(**case, out_value={'col': case['in_value'], 'errors_col': [None]})
    for case in [
        dict(in_dtype='integer', in_value=1, less_than_equal=1),
        dict(in_dtype='float', in_value=1.0, less_than_equal=1.0),
    ]
], *[
    dict(**case, out_value={'col': case['in_value'],
                            'errors_col': [{'type': 'less_than_equal', 'msg': case['error_msg']}]})
    for case in [
        dict(in_dtype='integer', in_value=1, less_than_equal=0,
             error_msg='Input should be less than or equal to 0'),
        dict(in_dtype='float', in_value=1.0, less_than_equal=0.0,
             error_msg='Input should be less than or equal to 0.0'),
    ]
]])
def test_less_than_equal(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.less_than_equal(case['less_than_equal'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_dtype='integer', in_value=2, multiple_of=1, out_value={'col': 2, 'errors_col': [None]}),
    dict(in_dtype='integer', in_value=2, multiple_of=3,
         out_value={'col': 2, 'errors_col': [{'type': 'multiple_of', 'msg': 'Input should be a multiple of 3'}]}),
    dict(in_dtype='integer', in_value=333, multiple_of=3, out_value={'col': 333, 'errors_col': [None]}),
    dict(in_dtype='integer', in_value=6, multiple_of=3, out_value={'col': 6, 'errors_col': [None]}),
])
def test_multiple_of(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', 'integer'))
    error_state = ErrorState('errors')
    rule = rules.multiple_of(case['multiple_of'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_dtype='string', in_value='1', out_value={'col': 1, 'col_bak': '1', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='1.0', out_value={'col': 1, 'col_bak': '1.0', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='foo', out_value={'col': None, 'col_bak': 'foo', 'errors_col': [
        {'type': 'int_parsing', 'msg': 'Input should be a valid integer, unable to parse value as an integer'}]}),
])
def test_int_parsing(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.int_parsing()

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_dtype='string', in_value='1', out_value={'col': 1.0, 'col_bak': '1', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='1.0', out_value={'col': 1.0, 'col_bak': '1.0', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='foo', out_value={'col': None, 'col_bak': 'foo', 'errors_col': [
        {'type': 'float_parsing', 'msg': 'Input should be a valid float, unable to parse value as a float'}]}),
])
def test_float_parsing(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.float_parsing()

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_dtype='integer', in_value=1, out_value={'col': '1', 'col_bak': 1, 'errors_col': []}),
    dict(in_dtype='float', in_value=1.0, out_value={'col': '1.0', 'col_bak': 1.0, 'errors_col': []}),
    dict(in_dtype='string', in_value='foo', out_value={'col': 'foo', 'col_bak': 'foo', 'errors_col': []}),
    dict(in_dtype='boolean', in_value=True, out_value={'col': 'true', 'col_bak': True, 'errors_col': []})
])
def test_string_parsing(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.string_parsing()

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_dtype='string', in_value='true', out_value={'col': True, 'col_bak': 'true', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='True', out_value={'col': True, 'col_bak': 'True', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='t', out_value={'col': True, 'col_bak': 't', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='false', out_value={'col': False, 'col_bak': 'false', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='False', out_value={'col': False, 'col_bak': 'False', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='f', out_value={'col': False, 'col_bak': 'f', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='yes', out_value={'col': True, 'col_bak': 'yes', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='no', out_value={'col': False, 'col_bak': 'no', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='Yes', out_value={'col': True, 'col_bak': 'Yes', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='No', out_value={'col': False, 'col_bak': 'No', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='Y', out_value={'col': True, 'col_bak': 'Y', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='N', out_value={'col': False, 'col_bak': 'N', 'errors_col': [None]}),
    dict(in_dtype='string', in_value='1', out_value={'col': True, 'col_bak': '1', 'errors_col': [None]}),
    dict(in_dtype='integer', in_value=1, out_value={'col': True, 'col_bak': 1, 'errors_col': [None]}),
    dict(in_dtype='integer', in_value=0, out_value={'col': False, 'col_bak': 0, 'errors_col': [None]}),
    dict(in_dtype='integer', in_value=100, out_value={'col': True, 'col_bak': 100, 'errors_col': [None]}),
    dict(in_dtype='float', in_value=1.5, out_value={'col': True, 'col_bak': 1.5, 'errors_col': [None]}),
    dict(in_dtype='string', in_value='foo', out_value={'col': None, 'col_bak': 'foo', 'errors_col': [
        {'type': 'boolean_parsing', 'msg': 'Input should be a valid boolean, unable to parse value as a boolean'}]}),
])
def test_boolean_parsing(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.boolean_parsing()

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_value='a', min_length=3, out_value={'col': 'a', 'errors_col': [
        {'type': 'min_length', 'msg': 'Input should have a minimum length of 3'}]}),
    dict(in_value='ab', min_length=3, out_value={'col': 'ab', 'errors_col': [
        {'type': 'min_length', 'msg': 'Input should have a minimum length of 3'}]}),
    dict(in_value='abc', min_length=3, out_value={'col': 'abc', 'errors_col': [None]}),
    dict(in_value='abcd', min_length=3, out_value={'col': 'abcd', 'errors_col': [None]}),
])
def test_min_length(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', 'string'))
    error_state = ErrorState('errors')
    rule = rules.min_length(case['min_length'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_value='ab', max_length=3, out_value={'col': 'ab', 'errors_col': [None]}),
    dict(in_value='abc', max_length=3, out_value={'col': 'abc', 'errors_col': [None]}),
    dict(in_value='abcd', max_length=3, out_value={'col': 'abcd', 'errors_col': [
        {'type': 'max_length', 'msg': 'Input should have a maximum length of 3'}]}),
    dict(in_value='abcde', max_length=3, out_value={'col': 'abcde', 'errors_col': [
        {'type': 'max_length', 'msg': 'Input should have a maximum length of 3'}]}),
])
def test_max_length(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', 'string'))
    error_state = ErrorState('errors')
    rule = rules.max_length(case['max_length'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_value='abc', pattern=r'^[0-9]+$', out_value={'col': 'abc', 'errors_col': [
        {'type': 'pattern', 'msg': 'Input should match the pattern ^[0-9]+$'}]}),
    dict(in_value='123', pattern=r'^[a-z]+$', out_value={'col': '123', 'errors_col': [
        {'type': 'pattern', 'msg': 'Input should match the pattern ^[a-z]+$'}]}),
    dict(in_value='123', pattern=r'^[0-9]+$', out_value={'col': '123', 'errors_col': [None]}),
    dict(in_value='abc', pattern=r'^[a-z]+$', out_value={'col': 'abc', 'errors_col': [None]}),
])
def test_pattern(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', 'string'))
    error_state = ErrorState('errors')
    rule = rules.pattern(case['pattern'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_value='abc', in_dtype='string', isin=['abcd', 'efgh'], out_value={
        'col': 'abc', 'errors_col': [{'type': 'isin', 'msg': "Input should be one of ['abcd', 'efgh']"}]}),
    dict(in_value='123', in_dtype='string', isin=[123, 456], out_value={'col': '123', 'errors_col': [None]}),
    dict(in_value=123, in_dtype='integer', isin=['123', '456'], out_value={'col': 123, 'errors_col': [None]}),
    dict(in_value='abc', in_dtype='string', isin=['abc', 'def'], out_value={'col': 'abc', 'errors_col': [None]})
])
def test_isin(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.isin(case['isin'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


@pytest.mark.parametrize('case', [
    dict(in_value='abc', in_dtype='string', notin=['abcd', 'efgh'], out_value={'col': 'abc', 'errors_col': [None]}),
    dict(in_value='123', in_dtype='string', notin=[123, 456], out_value={
        'col': '123', 'errors_col': [{'type': 'notin', 'msg': "Input should not be one of [123, 456]"}]}),
    dict(in_value=123, in_dtype='integer', notin=['123', '456'], out_value={
        'col': 123, 'errors_col': [{'type': 'notin', 'msg': "Input should not be one of ['123', '456']"}]}),
    dict(in_value='abc', in_dtype='string', notin=['abc', 'def'], out_value={
        'col': 'abc', 'errors_col': [{'type': 'notin', 'msg': "Input should not be one of ['abc', 'def']"}]})
])
def test_notin(spark_session, case):
    df = spark_session.createDataFrame(
        [(case['in_value'], [])], create_struct_type('col', case['in_dtype']))
    error_state = ErrorState('errors')
    rule = rules.notin(case['notin'])

    actual = rule.verify(df, 'col', error_state)

    assert actual.toJSON().map(json.loads).collect() == [case['out_value']]


def test_extra_forbidden(spark_session):
    df = spark_session.createDataFrame([('1', [])], create_struct_type('col1'))
    df = df.withColumn('col2', fn.lit('2'))
    error_state = ErrorState('errors')
    rule = rules.extra_forbidden({'col2'})

    actual = rule.verify(df, 'col1', error_state)

    assert actual.toJSON().map(json.loads).collect() == [
        {
            'col1': None,
            'col1_bak': '1',
            'col2': '2',
            'errors_col1': [
                {
                    'msg': 'Extra inputs are not permitted',
                    'type': 'extra_forbidden'
                }
            ]
        }
    ]


def test_extra_forbidden_allowed(spark_session):
    df = spark_session.createDataFrame([('1', [])], create_struct_type('col1'))
    df = df.withColumn('col2', fn.lit('2'))
    error_state = ErrorState('errors')
    rule = rules.extra_forbidden({'col1'})

    actual = rule.verify(df, 'col1', error_state)

    assert actual.toJSON().map(json.loads).collect() == [
        {
            'col1': '1',
            'col2': '2',
            'errors_col1': []
        }
    ]


def create_struct_type(col, dtype='string', error_col=None) -> StructType:
    return StructType.from_json(
        {
            'fields': [
                {
                    'name': col,
                    'type': dtype
                },
                {
                    'name': error_col or f'errors_{col}',
                    'type': {
                        'containsNull': False,
                        'elementType': 'void',
                        'type': 'array'
                    }
                }
            ],
            'type': 'struct'
        }
    )
