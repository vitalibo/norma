import pandas as pd

from norma.engines.pandas import rules


def test_error_state():
    error_state = rules.ErrorState(pd.RangeIndex(4))
    # iteration #1
    error_state.add_errors(pd.Series([True, False, False, False]), 'col1', details={'err': '#1'})

    assert error_state.errors == {
        0: {'col1': {'details': [{'err': '#1'}]}}}
    assert error_state.masks['col1'].equals(pd.Series([True, False, False, False]))

    # iteration #2
    error_state.add_errors(pd.Series([True, True, False, False]), 'col2', details={'err': '#2'})

    assert error_state.errors == {
        0: {'col1': {'details': [{'err': '#1'}]}, 'col2': {'details': [{'err': '#2'}]}},
        1: {'col2': {'details': [{'err': '#2'}]}}}
    assert error_state.masks['col1'].equals(pd.Series([True, False, False, False]))
    assert error_state.masks['col2'].equals(pd.Series([True, True, False, False]))

    # iteration #3
    error_state.add_errors(pd.Series([True, False, True, False]), 'col1', details={'err': '#2'})

    assert error_state.errors == {
        0: {'col1': {'details': [{'err': '#1'}, {'err': '#2'}]}, 'col2': {'details': [{'err': '#2'}]}},
        1: {'col2': {'details': [{'err': '#2'}]}},
        2: {'col1': {'details': [{'err': '#2'}]}}}

    assert error_state.masks['col1'].equals(pd.Series([True, False, True, False]))
    assert error_state.masks['col2'].equals(pd.Series([True, True, False, False]))

    # iteration #4
    error_state.add_errors(pd.Series([False, False, False, False]), 'col1', details={'err': '#3'})

    assert error_state.errors == {
        0: {'col1': {'details': [{'err': '#1'}, {'err': '#2'}]}, 'col2': {'details': [{'err': '#2'}]}},
        1: {'col2': {'details': [{'err': '#2'}]}},
        2: {'col1': {'details': [{'err': '#2'}]}}}

    assert len(error_state.masks) == 2
    assert error_state.masks['col1'].equals(pd.Series([True, False, True, False]))
    assert error_state.masks['col2'].equals(pd.Series([True, True, False, False]))
