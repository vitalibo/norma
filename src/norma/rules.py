from __future__ import annotations

import abc
from typing import Any, Iterable, TypeVar, Union

DataFrame = TypeVar('DataFrame')


class ErrorState:
    """
    Error state interface for data validation
    """

    def add_errors(self, boolmask, column, **kwargs):
        pass


class Rule(abc.ABC):
    """
    Rule interface for data validation
    """

    @abc.abstractmethod
    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        pass

    @classmethod
    def new(cls, func) -> Rule:
        """
        Create a new rule from a function
        """

        class _DecoratedRule(cls):
            def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
                return func(df, column, error_state)

        return _DecoratedRule()


class RuleProxy(Rule):
    """
    Rule proxy for data validation
    """

    def __init__(self, name: str, priority: float = 5, **kwargs):
        self.name = name
        self.priority = priority
        self.kwargs = kwargs

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        raise NotImplementedError('use engine-specific implementation')

    def __repr__(self):
        return f'{self.name}({", ".join([f"{k}={v}" for k, v in self.kwargs.items()])})'


def rule(func, priority: float = 5, **kwargs) -> Rule:
    return RuleProxy('rule', priority, func=func, **kwargs)


def required() -> Rule:
    return RuleProxy('required', 0)


def equal_to(eq: Any) -> Rule:
    return RuleProxy('equal_to', eq=eq)


def not_equal_to(ne: Any) -> Rule:
    return RuleProxy('not_equal_to', ne=ne)


def greater_than(gt: Any) -> Rule:
    return RuleProxy('greater_than', gt=gt)


def greater_than_equal(ge: Any) -> Rule:
    return RuleProxy('greater_than_equal', ge=ge)


def less_than(lt: Any) -> Rule:
    return RuleProxy('less_than', lt=lt)


def less_than_equal(le: Any) -> Rule:
    return RuleProxy('less_than_equal', le=le)


def multiple_of(multiple: Union[int, float]) -> Rule:
    return RuleProxy('multiple_of', multiple=multiple)


def min_length(value: int) -> Rule:
    return RuleProxy('min_length', value=value)


def max_length(value: int) -> Rule:
    return RuleProxy('max_length', value=value)


def pattern(regex: str) -> Rule:
    return RuleProxy('pattern', regex=regex)


def isin(values: Iterable[Any]) -> Rule:
    return RuleProxy('isin', values=values)


def notin(values: Iterable[Any]) -> Rule:
    return RuleProxy('notin', values=values)


def int_parsing() -> Rule:
    return RuleProxy('int_parsing', 1)


def float_parsing():
    return RuleProxy('float_parsing', 1)


def str_parsing() -> Rule:
    return RuleProxy('str_parsing', 1)


def bool_parsing() -> Rule:
    return RuleProxy('bool_parsing', 1)


def datetime_parsing() -> Rule:
    return RuleProxy('datetime_parsing', 1)


def date_parsing() -> Rule:
    return RuleProxy('date_parsing', 1)


def extra_forbidden(allowed: Iterable[str]) -> Rule:
    return RuleProxy('extra_forbidden', 0, allowed=allowed)
