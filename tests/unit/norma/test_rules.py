import numpy as np
import pandas as pd
import pytest

from norma import rules


@pytest.mark.parametrize('value, dtype', [
    (None, 'object'),
    (np.nan, 'object'),
    (pd.NA, 'boolean'),
    (np.nan, 'category'),
    (np.nan, 'float64'),
    (pd.NA, 'Int64'),
    (pd.NaT, 'datetime64[ns]'),
    (None, 'timedelta64[ns]'),
    (None, str),
])
def test_required_invalid(value, dtype):
    df = pd.DataFrame({'a': [value]}, dtype=dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.required()

    actual = rule.verify(df, column='a', error_state=error_state)

    assert actual is None
    assert error_state.masks['a'].equals(pd.Series([True]))
    assert error_state.errors[0]['a']['details'] == [{'type': 'missing', 'msg': 'Field required'}]


@pytest.mark.parametrize('value, dtype', [
    (1, 'int64'),
    (1.0, 'float64'),
    ('a', 'object'),
    (np.inf, 'float64'),
    (pd.Timestamp('2020-01-01'), 'datetime64[ns]'),
    (pd.Timedelta('1 days'), 'timedelta64[ns]'),
])
def test_required_valid(value, dtype):
    df = pd.DataFrame({'a': [value]}, dtype=dtype)
    error_state = rules.ErrorState(df.index)
    rule = rules.required()

    actual = rule.verify(df, column='a', error_state=error_state)

    assert actual is None
    assert error_state.masks['a'].equals(pd.Series([False]))
    assert error_state.errors[0] == {}

def test_int_parsing_valid():
    s1 = pd.Series([1, 2, 3], dtype='Int64')
    s2 = pd.Series(['1', '2N', '3'], dtype='str')
    s3 = pd.Series([1.2, 1.3, 3.4], dtype='float64')
    df = pd.DataFrame({'a': s1, 'b': s2, 'c': s3})

    # print(df.dtypes)
    # df = df.convert_dtypes()

    df['d'] =pd.to_numeric(df['a'], errors='coerce').astype('Float64')
    df['e'] =pd.to_numeric(df['b'], errors='coerce').astype('Float64')
    df['f'] =pd.to_numeric(df['c'].convert_dtypes(), errors='coerce').astype('Float64')

    print(df.dtypes)
    print(df)



