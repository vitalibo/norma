import numpy as np
import pandas as pd
import pytest

from norma import rules


def test_error_state_add_errors():
    error_state = rules.ErrorState(pd.RangeIndex(4))
    # iteration #1
    error_state.add_errors(pd.Series([True, False, False, False]), 'col1', {'err': '#1'})

    assert error_state.errors == {
        0: {'col1': {'details': [{'err': '#1'}]}}}
    assert error_state.masks['col1'].equals(pd.Series([True, False, False, False]))

    # iteration #2
    error_state.add_errors(pd.Series([True, True, False, False]), 'col2', {'err': '#2'})

    assert error_state.errors == {
        0: {'col1': {'details': [{'err': '#1'}]}, 'col2': {'details': [{'err': '#2'}]}},
        1: {'col2': {'details': [{'err': '#2'}]}}}
    assert error_state.masks['col1'].equals(pd.Series([True, False, False, False]))
    assert error_state.masks['col2'].equals(pd.Series([True, True, False, False]))

    # iteration #3
    error_state.add_errors(pd.Series([True, False, True, False]), 'col1', {'err': '#2'})

    assert error_state.errors == {
        0: {'col1': {'details': [{'err': '#1'}, {'err': '#2'}]}, 'col2': {'details': [{'err': '#2'}]}},
        1: {'col2': {'details': [{'err': '#2'}]}},
        2: {'col1': {'details': [{'err': '#2'}]}}}

    assert error_state.masks['col1'].equals(pd.Series([True, False, True, False]))
    assert error_state.masks['col2'].equals(pd.Series([True, True, False, False]))

    # iteration #4
    error_state.add_errors(pd.Series([False, False, False, False]), 'col1', {'err': '#3'})

    assert error_state.errors == {
        0: {'col1': {'details': [{'err': '#1'}, {'err': '#2'}]}, 'col2': {'details': [{'err': '#2'}]}},
        1: {'col2': {'details': [{'err': '#2'}]}},
        2: {'col1': {'details': [{'err': '#2'}]}}}

    assert len(error_state.masks) == 2
    assert error_state.masks['col1'].equals(pd.Series([True, False, True, False]))
    assert error_state.masks['col2'].equals(pd.Series([True, True, False, False]))


def assert_has_error(error_state, details):
    assert error_state.masks['col'].equals(pd.Series([True]))
    assert error_state.errors[0]['col']['details'] == details


def assert_no_error(error_state, details):
    assert error_state.masks['col'].equals(pd.Series([False]))
    assert error_state.errors[0] == {}


@pytest.mark.parametrize('value, dtype, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        (None, 'object'),
        (np.nan, 'object'),
        (pd.NA, 'boolean'),
        (np.nan, 'float64'),
        (pd.NA, 'Int64'),
        (pd.NaT, 'datetime64[ns]'),
        (None, 'timedelta64[ns]'),
        (None, str),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (1, 'int64'),
        (1.0, 'float64'),
        ('a', 'object'),
        (np.inf, 'float64'),
        (pd.Timestamp('2020-01-01'), 'datetime64[ns]'),
        (pd.Timedelta('1 days'), 'timedelta64[ns]'),
    ]
]])
def test_required(value, dtype, assert_error):
    df = pd.DataFrame({'col': [value]}, dtype=dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.required()

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'missing', 'msg': 'Field required'}])


@pytest.mark.parametrize('value, dtype, equal_to, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        (1, 'Int64', 2),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (1, 'Int64', 1),
    ]
]])
def test_equal_to(value, dtype, equal_to, assert_error):
    df = pd.DataFrame({'col': [value]})
    error_state = rules.ErrorState(df.index)
    rule = rules.equal_to(equal_to)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'equal_to', 'msg': f'Input should be equal to {equal_to}'}])


@pytest.mark.parametrize('value, dtype, not_equal_to, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        (1, 'Int64', 1),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (1, 'Int64', 2),
    ]
]])
def test_not_equal_to(value, dtype, not_equal_to, assert_error):
    df = pd.DataFrame({'col': [value]})
    error_state = rules.ErrorState(df.index)
    rule = rules.not_equal_to(not_equal_to)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'not_equal_to', 'msg': f'Input should not be equal to {not_equal_to}'}])


@pytest.mark.parametrize('value, dtype, greater_than, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        (1, 'Int64', 1),
        (0, 'Int64', 1),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (2, 'Int64', 1),
    ]
]])
def test_greater_than(value, dtype, greater_than, assert_error):
    df = pd.DataFrame({'col': [value]})
    error_state = rules.ErrorState(df.index)
    rule = rules.greater_than(greater_than)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'greater_than', 'msg': f'Input should be greater than {greater_than}'}])


@pytest.mark.parametrize('value, dtype, greater_than_equal, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        (0, 'Int64', 1),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (1, 'Int64', 1),
        (2, 'Int64', 1),
    ]
]])
def test_greater_than_equal(value, dtype, greater_than_equal, assert_error):
    df = pd.DataFrame({'col': [value]})
    error_state = rules.ErrorState(df.index)
    rule = rules.greater_than_equal(greater_than_equal)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [
        {'type': 'greater_than_equal', 'msg': f'Input should be greater than or equal to {greater_than_equal}'}
    ])


@pytest.mark.parametrize('value, dtype, less_than, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        (1, 'Int64', 1),
        (2, 'Int64', 1),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (0, 'Int64', 1),
    ]
]])
def test_less_than(value, dtype, less_than, assert_error):
    df = pd.DataFrame({'col': [value]})
    error_state = rules.ErrorState(df.index)
    rule = rules.less_than(less_than)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'less_than', 'msg': f'Input should be less than {less_than}'}])


@pytest.mark.parametrize('value, dtype, less_than_equal, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        (2, 'Int64', 1),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (1, 'Int64', 1),
        (0, 'Int64', 1),
    ]
]])
def test_less_than_equal(value, dtype, less_than_equal, assert_error):
    df = pd.DataFrame({'col': [value]})
    error_state = rules.ErrorState(df.index)
    rule = rules.less_than_equal(less_than_equal)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [
        {'type': 'less_than_equal', 'msg': f'Input should be less than or equal to {less_than_equal}'}
    ])


@pytest.mark.parametrize('value, dtype, multiple_of, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        (1, 'Int64', 2),
        (3, 'Int64', 2),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (4, 'Int64', 2),
    ]
]])
def test_multiple_of(value, dtype, multiple_of, assert_error):
    df = pd.DataFrame({'col': [value]})
    error_state = rules.ErrorState(df.index)
    rule = rules.multiple_of(multiple_of)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'multiple_of', 'msg': f'Input should be a multiple of {multiple_of}'}])


@pytest.mark.parametrize('in_value, in_dtype, out_value, out_dtype, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        ('foo', 'object', None, 'Int64'),
        ('', str, None, 'Int64'),
        ('one', 'object', None, 'Int64'),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (1, 'int64', 1, 'Int64'),
        (1.0, 'float64', 1, 'Int64'),
        (1.5, 'float64', 1, 'Int64'),
        (1.7, 'float64', 1, 'Int64'),
        (None, 'Int64', None, 'Int64'),
        ('1', 'object', 1, 'Int64'),
        ('1.0', 'object', 1, 'Int64'),
        (pd.Timestamp('2020-01-01'), 'datetime64[ns]', 1577836800000000000, 'Int64'),
        (True, 'boolean', 1, 'Int64'),
        (False, 'boolean', 0, 'Int64'),
    ]
]])
def test_int_parsing(in_value, in_dtype, out_value, out_dtype, assert_error):
    df = pd.DataFrame({'col': [in_value]}, dtype=in_dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.int_parsing()

    actual = rule.verify(df, column='col', error_state=error_state)

    pd.testing.assert_series_equal(actual, pd.Series([out_value], dtype=out_dtype, name='col'))
    assert_error(error_state, [
        {'type': 'int_parsing', 'msg': 'Input should be a valid integer, unable to parse value as an integer'}
    ])


@pytest.mark.parametrize('in_value, in_dtype, out_value, out_dtype, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        ('foo', 'object', None, 'Float64'),
        ('', str, None, 'Float64'),
        ('one', 'object', None, 'Float64'),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        (1, 'int64', 1.0, 'Float64'),
        (1.0, 'float64', 1.0, 'Float64'),
        (1.5, 'float64', 1.5, 'Float64'),
        (1.7, 'float64', 1.7, 'Float64'),
        (None, 'Int64', None, 'Float64'),
        ('1', 'object', 1.0, 'Float64'),
        ('1.0', 'object', 1.0, 'Float64'),
        (pd.Timestamp('2020-01-01'), 'datetime64[ns]', 1577836800000000000.0, 'Float64'),
        (True, 'boolean', 1.0, 'Float64'),
        (False, 'boolean', 0.0, 'Float64'),
    ]
]])
def test_float_parsing(in_value, in_dtype, out_value, out_dtype, assert_error):
    df = pd.DataFrame({'col': [in_value]}, dtype=in_dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.float_parsing()

    actual = rule.verify(df, column='col', error_state=error_state)

    pd.testing.assert_series_equal(actual, pd.Series([out_value], dtype=out_dtype, name='col'))
    assert_error(error_state, [
        {'type': 'float_parsing', 'msg': 'Input should be a valid float, unable to parse value as a float'}
    ])


@pytest.mark.parametrize('in_value, in_dtype, out_value, assert_error', [*[
    (*params, assert_no_error)
    for params in [
        (1, 'int64', '1'),
        (1.5, 'float64', '1.5'),
        ('foo', 'object', 'foo'),
        (None, 'Int64', None),
        ('', str, ''),
        (np.nan, 'float64', None),
        (pd.NA, 'Int64', None),
        (pd.NaT, 'datetime64[ns]', None),
        (True, 'boolean', 'True'),
        ({'foo': 'bar'}, 'object', "{'foo': 'bar'}"),
    ]
]])
def test_string_parsing(in_value, in_dtype, out_value, assert_error):
    df = pd.DataFrame({'col': [in_value]}, dtype=in_dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.string_parsing()

    actual = rule.verify(df, column='col', error_state=error_state)

    pd.testing.assert_series_equal(actual, pd.Series([out_value], dtype="string[python]", name='col'))
    assert_error(error_state, [])


@pytest.mark.parametrize('in_value, in_dtype, out_value, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        ('foo', 'object', None),
        ('', str, None),
        ({'foo': 'bar'}, 'object', None),
        ('truetrue', 'object', None),
        ('falsetrue', 'object', None),
        ('yesyes', 'object', None),
        ('tt', 'object', None),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        ('true', 'object', True),
        ('True', 'object', True),
        ('t', 'object', True),
        ('false', 'object', False),
        ('False', 'object', False),
        ('f', 'object', False),
        ('yes', 'object', True),
        ('Yes', 'object', True),
        ('y', 'object', True),
        ('no', 'object', False),
        ('No', 'object', False),
        ('n', 'object', False),
        (1, 'int64', True),
        (1.5, 'float64', True),
        ('1.5', 'object', True),
        (0, 'int64', False),
        (100, 'int64', True),
        (None, 'Int64', None),
        (np.nan, 'float64', None),
        (pd.NA, 'Int64', None),
        (True, 'boolean', True),
        (pd.NaT, 'datetime64[ns]', None),
    ]
]])
def test_boolean_parsing(in_value, in_dtype, out_value, assert_error):
    df = pd.DataFrame({'col': [in_value]}, dtype=in_dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.boolean_parsing()

    actual = rule.verify(df, column='col', error_state=error_state)

    pd.testing.assert_series_equal(actual, pd.Series([out_value], dtype='boolean', name='col'))
    assert_error(error_state, [
        {'type': 'boolean_parsing', 'msg': 'Input should be a valid boolean, unable to parse value as a boolean'}
    ])


@pytest.mark.parametrize('value, dtype, threshold, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        ('a', 'object', 3),
        ('ab', 'string[python]', 3),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        ('abc', 'string[python]', 3),
        ('abcd', 'string[python]', 2),
    ]
]])
def test_min_length(value, dtype, threshold, assert_error):
    df = pd.DataFrame({'col': [value]}, dtype=dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.min_length(threshold)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'min_length', 'msg': f'Input should have a minimum length of {threshold}'}])


@pytest.mark.parametrize('value, dtype, threshold, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        ('abc', 'string[python]', 2),
        ('abcd', 'string[python]', 3),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        ('a', 'object', 3),
        ('ab', 'string[python]', 3),
    ]
]])
def test_max_length(value, dtype, threshold, assert_error):
    df = pd.DataFrame({'col': [value]}, dtype=dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.max_length(threshold)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'max_length', 'msg': f'Input should have a maximum length of {threshold}'}])


@pytest.mark.parametrize('value, dtype, regex, assert_error', [*[
    (*params, assert_has_error)
    for params in [
        ('abc', 'object', r'^[0-9]+$'),
        ('123', 'string[python]', r'^[a-z]+$'),
    ]
], *[
    (*params, assert_no_error)
    for params in [
        ('123', 'string[python]', r'^[0-9]+$'),
        ('abc', 'object', r'^[a-z]+$'),
    ]
]])
def test_pattern(value, dtype, regex, assert_error):
    df = pd.DataFrame({'col': [value]}, dtype=dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.pattern(regex)

    actual = rule.verify(df, column='col', error_state=error_state)

    assert actual is not None
    assert_error(error_state, [{'type': 'pattern', 'msg': f'Input should match the pattern {regex}'}])
