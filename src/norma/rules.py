from __future__ import annotations

import abc
import re
from typing import Any, Iterable, Optional, TypeVar, Union

DataFrame = TypeVar('DataFrame')


class ErrorState(abc.ABC):
    """
    Interface for managing error states during data validation
    """

    @abc.abstractmethod
    def add_errors(self, boolmask, column, **kwargs):
        """
        Add errors to the error state for a given column

        :param boolmask: Boolean expression for the error condition (True values indicate errors)
        :param column: The name of the column where the errors occurred
        :param kwargs: Additional details about the errors
        """


class Rule(abc.ABC):
    """
    Rule interface for data validation rules
    """

    @abc.abstractmethod
    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> Optional[DataFrame]:
        """
        Verify a column in a DataFrame

        :param df: The DataFrame containing the data to be verified
        :param column: The name of the column to be verified
        :param error_state: The error state object to store error information
        :return: The DataFrame with validated data of None if errors occurred
        """

    @classmethod
    def new(cls, func) -> Rule:
        """
        Create a new rule from a function

        :param func: The function is to decorate and used as a rule
        """

        class _DecoratedRule(cls):
            def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
                return func(df, column, error_state)

        return _DecoratedRule()


class RuleProxy(Rule):
    """
    Rule proxy acts as a placeholder for a rule that will be implemented by an engine-specific rule

    :param name: The name of the rule
    :param priority: The priority of the rule (lower values are applied first)
    :param kwargs: Additional keyword arguments to be passed to the rule
    """

    def __init__(self, name: str, priority: float = 5, **kwargs):
        self.name = name
        self.priority = priority
        self.kwargs = kwargs

    def verify(self, df: DataFrame, column: str, error_state: ErrorState) -> DataFrame:
        raise NotImplementedError('use engine-specific implementation')


def rule(func, priority: float = 5, **kwargs) -> Rule:
    """
    Create a new rule from a function

    :param func: The function to be used as the rule
    :param priority: The priority of applying the rule (lower values are applied first, default is 5)
    :param kwargs: Additional keyword arguments to be passed to the rule
    :return: A new rule object
    """

    return RuleProxy('rule', priority, func=func, **kwargs)


def required() -> Rule:
    """
    Ensure that the values are present and not null
    """

    return RuleProxy('required', 0)


def equal_to(eq: Any) -> Rule:
    """
    Ensure that the values are equal to the specified value

    :param eq: The value to compare against
    """

    if eq is None:
        raise ValueError('comparison value must not be None')

    return RuleProxy('equal_to', eq=eq)


def not_equal_to(ne: Any) -> Rule:
    """
    Ensure that the values are not equal to the specified value

    :param ne: The value to compare against
    """

    if ne is None:
        raise ValueError('comparison value must not be None')

    return RuleProxy('not_equal_to', ne=ne)


def greater_than(gt: Any) -> Rule:
    """
    Ensure that the values are greater than the specified value

    :param gt: The value to compare against
    """

    if gt is None:
        raise ValueError('comparison value must not be None')

    return RuleProxy('greater_than', gt=gt)


def greater_than_equal(ge: Any) -> Rule:
    """
    Ensure that the values are greater than or equal to the specified value

    :param ge: The value to compare against
    """

    if ge is None:
        raise ValueError('comparison value must not be None')

    return RuleProxy('greater_than_equal', ge=ge)


def less_than(lt: Any) -> Rule:
    """
    Ensure that the values are less than the specified value

    :param lt: The value to compare against
    """

    if lt is None:
        raise ValueError('comparison value must not be None')

    return RuleProxy('less_than', lt=lt)


def less_than_equal(le: Any) -> Rule:
    """
    Ensure that the values are less than or equal to the specified value

    :param le: The value to compare against
    """

    if le is None:
        raise ValueError('comparison value must not be None')

    return RuleProxy('less_than_equal', le=le)


def multiple_of(multiple: Union[int, float]) -> Rule:
    """
    Ensure that the values are a multiple of the specified value

    :param multiple: The value that the data should be a multiple of (must be an integer or a float)
    """

    if multiple is None:
        raise ValueError('multiple_of must not be None')
    if not isinstance(multiple, (int, float)):
        raise ValueError('multiple_of must be an integer or a float')

    return RuleProxy('multiple_of', multiple=multiple)


def min_length(value: int) -> Rule:
    """
    Ensure that the length of the values is at least the specified value

    :param value: The minimum length required
    """

    if not isinstance(value, int):
        raise ValueError('min_length must be an integer')
    if value < 0:
        raise ValueError('min_length must be a non-negative integer')

    return RuleProxy('min_length', value=value)


def max_length(value: int) -> Rule:
    """
    Ensure that the length of the values is at most the specified value

    :param value: The maximum length allowed
    """

    if not isinstance(value, int):
        raise ValueError('max_length must be an integer')
    if value < 0:
        raise ValueError('max_length must be a non-negative integer')

    return RuleProxy('max_length', value=value)


def pattern(regex: str) -> Rule:
    """
    Ensure that the values match the specified regular expression pattern

    :param regex: The regular expression pattern to match
    """

    if not isinstance(regex, str):
        raise ValueError('pattern must be a string')
    try:
        re.compile(regex)
    except re.error as e:
        raise ValueError('pattern must be a valid regular expression') from e

    return RuleProxy('pattern', regex=regex)


def isin(values: Iterable[Any]) -> Rule:
    """
    Ensure that the values are within the specified iterable

    :param values: The iterable containing the allowed values (must be a list, tuple, or set)
    """

    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return RuleProxy('isin', values=values)


def notin(values: Iterable[Any]) -> Rule:
    """
    Ensure that the values are not within the specified iterable

    :param values: The iterable containing the disallowed values (must be a list, tuple, or set)
    """

    if not isinstance(values, (list, tuple, set)):
        raise ValueError('values must be a list, tuple, or set')

    return RuleProxy('notin', values=values)


def unique_items() -> Rule:
    """
    Ensure that the values are unique within the array column
    """

    return RuleProxy('unique_items')


def max_items(value: int) -> Rule:
    """
    Ensure that the number of items in the array is at most the specified value

    :param value: The maximum number of items allowed
    """

    if not isinstance(value, int):
        raise ValueError('max_items must be an integer')
    if value < 0:
        raise ValueError('max_items must be a non-negative integer')

    return RuleProxy('max_items', value=value)


def min_items(value: int) -> Rule:
    """
    Ensure that the number of items in the array is at least the specified value

    :param value: The minimum number of items required
    """

    if not isinstance(value, int):
        raise ValueError('min_items must be an integer')
    if value < 0:
        raise ValueError('min_items must be a non-negative integer')

    return RuleProxy('min_items', value=value)


def int_parsing() -> Rule:
    """
    Ensure that the values are integers or parse as integers if possible
    :return:
    """

    return RuleProxy('int_parsing', 1)


def float_parsing():
    """
    Ensure that the values are floats or parse as floats if possible
    """

    return RuleProxy('float_parsing', 1)


def str_parsing() -> Rule:
    """
    Ensure that the values are strings or parse as strings if possible
    """

    return RuleProxy('str_parsing', 1)


def bool_parsing() -> Rule:
    """
    Ensure that the values are booleans or parse as booleans if possible
    """

    return RuleProxy('bool_parsing', 1)


def datetime_parsing() -> Rule:
    """
    Ensure that the values are date-times or parse as date-times if possible
    """

    return RuleProxy('datetime_parsing', 1)


def date_parsing() -> Rule:
    """
    Ensure that the values are dates or parse as dates if possible
    """

    return RuleProxy('date_parsing', 1)


def time_parsing() -> Rule:
    """
    Ensure that the values are times or parse as times if possible
    """

    return RuleProxy('time_parsing', 1)


def duration_parsing() -> Rule:
    """
    Ensure that the values are durations or parse as durations if possible
    """

    return RuleProxy('duration_parsing', 1)


def uuid_parsing() -> Rule:
    """
    Ensure that the values are valid UUIDs
    """

    return RuleProxy('uuid_parsing', 1)


def ipv4_address() -> Rule:
    """
    Ensure that the values are valid IPv4 addresses
    """

    return RuleProxy('ipv4_address', 1)


def ipv6_address() -> Rule:
    """
    Ensure that the values are valid IPv6 addresses
    """

    return RuleProxy('ipv6_address', 1)


def uri_parsing() -> Rule:
    """
    Ensure that the values are valid URIs
    """

    return RuleProxy('uri_parsing', 1)


def object_parsing(schema) -> Rule:
    """
    Ensure that the values are objects or parse as objects if possible
    """

    return RuleProxy('object_parsing', 1, schema=schema)


def array_parsing(schema) -> Rule:
    """
    Ensure that the values are arrays or parse as arrays if possible
    """

    return RuleProxy('array_parsing', 1, schema=schema)
