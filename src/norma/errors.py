from collections import UserDict, UserString
from types import MappingProxyType
from typing import Any


class ErrorDetails(UserDict):
    """
    A class to represent the details of an error.
    """

    def __init__(self, type: str, msg: str):  # noqa pylint: disable=redefined-builtin
        super().__init__()
        self.data = MappingProxyType({'type': type, 'msg': msg})

    @property
    def type(self):
        return self.data['type']

    @property
    def message(self):
        return ErrorDetailMessage(self.data['msg'])

    def format(self, *args: Any, **kwds: Any):
        return {'type': self.type, 'msg': self.message.format(*args, **kwds)}


class ErrorDetailMessage(UserString):
    """
    A class to represent the message of an error.
    """

    def format(self, *args: Any, **kwds: Any) -> str:  # pylint: disable=arguments-differ)
        def escape(v):
            if isinstance(v, str):
                return f'"{v}"'
            if isinstance(v, bool):
                return 'true' if v else 'false'
            if isinstance(v, (list, tuple)):
                if len(v) == 1:
                    return escape(v[0])
                return ', '.join([escape(vv) for vv in v[:-1]]) + f' or {escape(v[-1])}'
            return str(v)

        return self.data.format(
            *[escape(v) for v in args],
            **{k: (v if k.startswith('_') else escape(v)) for k, v in kwds.items()}
        )


MISSING = ErrorDetails('missing', 'Field required')
EXTRA_FORBIDDEN = ErrorDetails('extra_forbidden', 'Extra inputs are not permitted')
EQUAL_TO = ErrorDetails('equal_to', 'Input should be equal to {eq}')
NOT_EQUAL_TO = ErrorDetails('not_equal_to', 'Input should not be equal to {ne}')
GREATER_THAN = ErrorDetails('greater_than', 'Input should be greater than {gt}')
GREATER_THAN_EQUAL = ErrorDetails('greater_than_equal', 'Input should be greater than or equal to {ge}')
LESS_THAN = ErrorDetails('less_than', 'Input should be less than {lt}')
LESS_THAN_EQUAL = ErrorDetails('less_than_equal', 'Input should be less than or equal to {le}')
MULTIPLE_OF = ErrorDetails('multiple_of', 'Input should be a multiple of {multiple_of}')
STRING_TOO_SHORT = ErrorDetails(
    'string_too_short', 'String should have at least {min_length} character{_expected_plural_}')
STRING_TOO_LONG = ErrorDetails(
    'string_too_long', 'String should have at most {max_length} character{_expected_plural_}')
STRING_PATTERN_MISMATCH = ErrorDetails('string_pattern_mismatch', 'String should match pattern {pattern}')
ENUM = ErrorDetails('enum', 'Input should be {expected}')
NOT_ENUM = ErrorDetails('not_enum', 'Input should not be {unexpected}')
