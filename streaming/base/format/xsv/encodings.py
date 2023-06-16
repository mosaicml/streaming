# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Encode and Decode samples in a supported Tabular format."""

from abc import ABC, abstractmethod
from typing import Any

__all__ = ['is_xsv_encoding', 'xsv_decode', 'xsv_encode']


class Encoding(ABC):
    """XSV (e.g. CSV, TSV) types."""

    @classmethod
    @abstractmethod
    def encode(cls, obj: Any) -> str:
        """Encode the given data from the original object to string.

        Args:
            obj (Any): Decoded object.

        Returns:
            str: Encoded data in string form.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def decode(cls, obj: str) -> Any:
        """Decode the given data from string to the original object.

        Args:
            obj (str): Encoded data in string form.

        Returns:
            Any: Decoded object.
        """
        raise NotImplementedError

    @staticmethod
    def _validate(data: Any, expected_type: Any) -> None:
        if not isinstance(data, expected_type):
            raise AttributeError(
                f'data should be of type {expected_type}, but instead, found as {type(data)}')


class Str(Encoding):
    """Store str."""

    @classmethod
    def encode(cls, obj: Any) -> str:
        cls._validate(obj, str)
        return obj

    @classmethod
    def decode(cls, obj: str) -> Any:
        return obj


class Int(Encoding):
    """Store int."""

    @classmethod
    def encode(cls, obj: Any) -> str:
        cls._validate(obj, int)
        return str(obj)

    @classmethod
    def decode(cls, obj: str) -> Any:
        return int(obj)


class Float(Encoding):
    """Store float."""

    @classmethod
    def encode(cls, obj: Any) -> str:
        cls._validate(obj, float)
        return str(obj)

    @classmethod
    def decode(cls, obj: str) -> Any:
        return float(obj)


_encodings = {'str': Str, 'int': Int, 'float': Float}


def is_xsv_encoding(encoding: str) -> bool:
    """Get whether the given encoding is supported.

    Args:
        encoding (str): Encoding.

    Returns:
        bool: Whether encoding is supported.
    """
    return encoding in _encodings


def xsv_encode(encoding: str, value: Any) -> str:
    """Encode the given data from the original object to string.

    Args:
        encoding (str): Encoding name.
        value (Any): Object to encode.

    Returns:
        str: Data in string form.
    """
    cls = _encodings[encoding]
    return cls.encode(value)


def xsv_decode(encoding: str, value: str) -> Any:
    """Decode the given data from string to the original object.

    Args:
        encoding (str): Encoding name.
        value (str): Object to decode.

    Returns:
        Any: Decoded object.
    """
    cls = _encodings[encoding]
    return cls.decode(value)
