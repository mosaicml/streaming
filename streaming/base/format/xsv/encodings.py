# Copyright 2022 MosaicML Streaming authors
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
        """Encode the object.

        Args:
            obj (Any): The object.

        Returns:
            str: String form.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def decode(cls, obj: str) -> Any:
        """Decode the object.

        Args:
            obj (str): String form.

        Returns:
            Any: The object.
        """
        raise NotImplementedError


class Str(Encoding):
    """Store str."""

    @classmethod
    def encode(cls, obj: Any) -> str:
        assert isinstance(obj, str)
        return obj

    @classmethod
    def decode(cls, obj: str) -> Any:
        return obj


class Int(Encoding):
    """Store int."""

    @classmethod
    def encode(cls, obj: Any) -> str:
        assert isinstance(obj, int)
        return str(obj)

    @classmethod
    def decode(cls, obj: str) -> Any:
        return int(obj)


class Float(Encoding):
    """Store float."""

    @classmethod
    def encode(cls, obj: Any) -> str:
        assert isinstance(obj, float)
        return str(obj)

    @classmethod
    def decode(cls, obj: str) -> Any:
        return float(obj)


_encodings = {'str': Str, 'int': Int, 'float': Float}


def is_xsv_encoding(encoding: str) -> bool:
    """Get whether this is a supported encoding.

    Args:
        encoding (str): Encoding.

    Returns:
        bool: Whether encoding is supported.
    """
    return encoding in _encodings


def xsv_encode(encoding: str, value: Any) -> str:
    """Encode the object.

    Args:
        encoding (str): The encoding.
        value (Any): The object.

    Returns:
        str: String form.
    """
    cls = _encodings[encoding]
    return cls.encode(value)


def xsv_decode(encoding: str, value: str) -> Any:
    """Encode the object.

    Args:
        encoding (str): The encoding.
        value (str): String form.

    Returns:
        Any: The object.
    """
    cls = _encodings[encoding]
    return cls.decode(value)
