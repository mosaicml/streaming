# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether sample encoding is of supported JSONL types."""

from abc import ABC, abstractmethod
from typing import Any

from streaming.format.base.type import Float as LogicalFloat
from streaming.format.base.type import Int as LogicalInt
from streaming.format.base.type import Str as LogicalStr
from streaming.format.base.type import Type as LogicalType

__all__ = ['is_jsonl_encoded', 'is_jsonl_encoding', 'jsonl_encoding_to_logical_type']


class Encoding(ABC):
    """Encoding of an object of JSONL type."""

    @classmethod
    @abstractmethod
    def is_encoded(cls, obj: Any) -> bool:
        """Get whether the given object is of this type.

        Args:
            obj (Any): Encoded object.

        Returns:
            bool: Whether of this type.
        """
        raise NotImplementedError

    @staticmethod
    def _validate(data: Any, expected_type: Any) -> bool:
        if not isinstance(data, expected_type):
            raise AttributeError(
                f'data should be of type {expected_type}, but instead, found as {type(data)}')
        return True


class Str(Encoding):
    """Store str."""

    logical_type = LogicalStr

    @classmethod
    def is_encoded(cls, obj: Any) -> bool:
        return cls._validate(obj, str)


class Int(Encoding):
    """Store int."""

    logical_type = LogicalInt

    @classmethod
    def is_encoded(cls, obj: Any) -> bool:
        return cls._validate(obj, int)


class Float(Encoding):
    """Store float."""

    logical_type = LogicalFloat

    @classmethod
    def is_encoded(cls, obj: Any) -> bool:
        return cls._validate(obj, float)


_encodings = {
    'str': Str,
    'int': Int,
    'float': Float,
}


def is_jsonl_encoded(encoding: str, value: Any) -> bool:
    """Get whether the given object is of this encoding type.

    Args:
        encoding (str): The encoding.
        value (Any): The object.

    Returns:
        bool: Whether of this type.
    """
    cls = _encodings[encoding]
    return cls.is_encoded(value)


def is_jsonl_encoding(encoding: str) -> bool:
    """Get whether the given encoding is supported.

    Args:
        encoding (str): Encoding.

    Returns:
        bool: Whether encoding is supported.
    """
    return encoding in _encodings


def jsonl_encoding_to_logical_type(encoding: str) -> LogicalType:
    """Get the logical type of the given encoding.

    Args:
        encoding (str): Encoding.

    Returns:
        LogicalType: Its logical type.
    """
    cls = _encodings[encoding]
    return cls.logical_type()
