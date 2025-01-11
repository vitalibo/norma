from typing import Optional

import pandas as pd


def required(col: str):
    def rule(df, error_state):
        error_mask = df[col].isna()

        error_state.add_errors(error_mask, col, {
            'type': 'missing',
            'msg': 'Field required'
        })

    return rule


def int_parsing(col: str):
    def rule(df, error_state):
        numeric_df = pd.to_numeric(df[col], errors='coerce')

        error_mask = numeric_df.isna() & df[col].notna()

        error_state.add_errors(error_mask, col, {
            'type': 'int_parsing',
            'msg': 'Input should be a valid integer, unable to parse'
        })

        return numeric_df

    return rule


def equal(col: str, eq: int):
    def rule(df, error_state):
        error_mask = df[col][df[col].notna()] != eq

        error_state.add_errors(error_mask, col, {
            'type': 'equal',
            'msg': f'Input should be equal to {eq}'
        })

    return rule


def not_equal(col: str, ne: int):
    def rule(df, error_state):
        error_mask = df[col][df[col].notna()] == ne

        error_state.add_errors(error_mask, col, {
            'type': 'not_equal',
            'msg': f'Input should not be equal to {ne}'
        })

    return rule


def greater_than(col: str, gt: int):
    def rule(df, error_state):
        error_mask = df[col][df[col].notna()] <= gt

        error_state.add_errors(error_mask, col, {
            'type': 'greater_than',
            'msg': f'Input should be greater than {gt}'
        })

    return rule


def greater_than_equal(col: str, ge: int):
    def rule(df, error_state):
        error_mask = df[col][df[col].notna()] < ge

        error_state.add_errors(error_mask, col, {
            'type': 'greater_than_equal',
            'msg': f'Input should be greater than or equal to {ge}'
        })

    return rule


def less_than(col: str, lt: int):
    def rule(df, error_state):
        error_mask = df[col][df[col].notna()] >= lt

        error_state.add_errors(error_mask, col, {
            'type': 'less_than',
            'msg': f'Input should be less than {lt}'
        })

    return rule


def less_than_equal(col: str, le: int):
    def rule(df, error_state):
        error_mask = df[col][df[col].notna()] > le

        error_state.add_errors(error_mask, col, {
            'type': 'less_than_equal',
            'msg': f'Input should be less than or equal to {le}'
        })

    return rule


def verify_number(
        col: str,
        nullable: bool = False,
        eq: Optional[float] = None,
        ne: Optional[float] = None,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
):
    def __rules(**kwargs):
        rules_def = {
            'eq': equal,
            'ne': not_equal,
            'gt': greater_than,
            'ge': greater_than_equal,
            'lt': less_than,
            'le': less_than_equal,
        }

        return [rules_def[k](col, v) for k, v in kwargs.items() if v is not None]

    rules = [
        *([required(col)] if not nullable else []),
        int_parsing(col),
        *__rules(eq=eq, ne=ne, gt=gt, ge=ge, lt=lt, le=le),
    ]

    def rule(df, error_state):
        for r in rules:
            new_df = r(df, error_state)
            if new_df is not None:
                df[col] = new_df

    return rule
