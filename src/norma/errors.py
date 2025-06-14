from collections import UserDict, UserString
from types import MappingProxyType
from typing import Any


class Details(UserDict):
    """
    Represents a detailed error message for a specific validation error
    """

    def __init__(self, detail_type: str, detail_msg: str):
        super().__init__()
        self.data = MappingProxyType({'type': detail_type, 'msg': detail_msg})

    @property
    def type(self):
        """
        Gets the type of the error
        """

        return self.data['type']

    @property
    def message(self):
        """
        Gets the message of the error
        """

        return Message(self.data['msg'])

    def format(self, *args: Any, **kwds: Any):
        """
        Formats the error message with the given arguments
        """

        return {'type': self.type, 'msg': self.message.format(*args, **kwds)}


class Message(UserString):
    """
    Represents the message of a detailed error
    """

    def format(self, *args: Any, **kwds: Any) -> str:  # pylint: disable=arguments-differ
        """
        Formats the message with the given arguments
        """

        def escape(v):
            if isinstance(v, str):
                return f'"{v}"'
            if isinstance(v, bool):
                return 'true' if v else 'false'
            if isinstance(v, (list, tuple, set)):
                if len(v) == 1:
                    return escape(v[0])
                return ', '.join([escape(vv) for vv in v[:-1]]) + f' or {escape(v[-1])}'
            return str(v)

        return self.data.format(
            *[escape(v) for v in args],
            **{k: (v if k.startswith('_') else escape(v)) for k, v in kwds.items()}
        )


MISSING = Details('missing', 'Field required')
EXTRA_FORBIDDEN = Details('extra_forbidden', 'Extra inputs are not permitted')
EQUAL_TO = Details('equal_to', 'Input should be equal to {eq}')
NOT_EQUAL_TO = Details('not_equal_to', 'Input should not be equal to {ne}')
GREATER_THAN = Details('greater_than', 'Input should be greater than {gt}')
GREATER_THAN_EQUAL = Details('greater_than_equal', 'Input should be greater than or equal to {ge}')
LESS_THAN = Details('less_than', 'Input should be less than {lt}')
LESS_THAN_EQUAL = Details('less_than_equal', 'Input should be less than or equal to {le}')
MULTIPLE_OF = Details('multiple_of', 'Input should be a multiple of {multiple_of}')
STRING_TOO_SHORT = Details('string_too_short', 'String should have at least {min_length} character{_plural_}')
STRING_TOO_LONG = Details('string_too_long', 'String should have at most {max_length} character{_plural_}')
STRING_PATTERN_MISMATCH = Details('string_pattern_mismatch', 'String should match pattern {pattern}')
ENUM = Details('enum', 'Input should be {expected}')
NOT_ENUM = Details('not_enum', 'Input should not be {unexpected}')
BOOL_TYPE = Details('bool_type', 'Input should be a valid boolean')
BOOL_PARSING = Details('bool_parsing', 'Input should be a valid boolean, unable to interpret input')
INT_TYPE = Details('int_type', 'Input should be a valid integer')
INT_PARSING = Details('int_parsing', 'Input should be a valid integer, unable to parse string as an integer')
FLOAT_TYPE = Details('float_type', 'Input should be a valid number')
FLOAT_PARSING = Details('float_parsing', 'Input should be a valid number, unable to parse string as a number')
STRING_TYPE = Details('string_type', 'Input should be a valid string')
DATE_TYPE = Details('date_type', 'Input should be a valid date')
DATE_PARSING = Details('date_parsing', 'Input should be a valid date, unable to parse string as a date')
DATETIME_TYPE = Details('datetime_type', 'Input should be a valid datetime')
DATETIME_PARSING = Details('datetime_parsing', 'Input should be a valid datetime, unable to parse string as a datetime')
OBJECT_TYPE = Details('object_type', 'Input should be a valid object')
OBJECT_PARSING = Details('object_parsing', 'Input should be a valid object, unable to parse string as an object')
UUID_TYPE = Details('uuid_type', 'UUID input should be a string')
UUID_PARSING = Details('uuid_parsing', 'Input should be a valid UUID, unable to parse string as a UUID')
ARRAY_TYPE = Details('array_type', 'Input should be a valid array')
ARRAY_PARSING = Details('array_parsing', 'Input should be a valid array, unable to parse string as an array')
TOO_SHORT = Details('too_short', '{_type_} should have at least {min_length} item{_plural_}')
TOO_LONG = Details('too_long', '{_type_} should have at most {max_length} item{_plural_}')
IPV4 = Details('ipv4', 'Input is not a valid IPv4 address')
IPV6 = Details('ipv6', 'Input is not a valid IPv6 address')
