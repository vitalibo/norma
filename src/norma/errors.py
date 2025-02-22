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
STRING_TOO_SHORT = ErrorDetails('string_too_short', 'String should have at least {min_length} character{_plural_}')
STRING_TOO_LONG = ErrorDetails('string_too_long', 'String should have at most {max_length} character{_plural_}')
STRING_PATTERN_MISMATCH = ErrorDetails('string_pattern_mismatch', 'String should match pattern {pattern}')
ENUM = ErrorDetails('enum', 'Input should be {expected}')
NOT_ENUM = ErrorDetails('not_enum', 'Input should not be {unexpected}')
BOOL_TYPE = ErrorDetails('bool_type', 'Input should be a valid boolean')
BOOL_PARSING = ErrorDetails('bool_parsing', 'Input should be a valid boolean, unable to interpret input')
INT_TYPE = ErrorDetails('int_type', 'Input should be a valid integer')
INT_PARSING = ErrorDetails('int_parsing', 'Input should be a valid integer, unable to parse string as an integer')
FLOAT_TYPE = ErrorDetails('float_type', 'Input should be a valid number')
FLOAT_PARSING = ErrorDetails('float_parsing', 'Input should be a valid number, unable to parse string as a number')
STRING_TYPE = ErrorDetails('string_type', 'Input should be a valid string')
DATE_TYPE = ErrorDetails('date_type', 'Input should be a valid date')
DATE_PARSING = ErrorDetails('date_parsing', 'Input should be a valid date, unable to parse string as a date')
DATETIME_TYPE = ErrorDetails('datetime_type', 'Input should be a valid datetime')
DATETIME_PARSING = ErrorDetails('datetime_parsing',
                                'Input should be a valid datetime, unable to parse string as a datetime')
