from typing import Any

__all__ = ['is_json_encoded', 'is_json_encoding']


class Encoding(object):
    """JSON types."""

    @classmethod
    def is_encoded(cls, obj: Any) -> bool:
        """Get whether the given object is of this type.

        Args:
            obj (Any): The object.

        Returns:
            bool: Whether of this type.
        """
        raise NotImplementedError


class Str(Encoding):
    """Store str."""

    @classmethod
    def is_encoded(cls, obj: Any) -> bool:
        return isinstance(obj, str)


class Int(Encoding):
    """Store int."""

    @classmethod
    def is_encoded(cls, obj: Any) -> bool:
        return isinstance(obj, int)


class Float(Encoding):
    """Store float."""

    @classmethod
    def is_encoded(cls, obj: Any) -> bool:
        return isinstance(obj, float)


_encodings = {'str': Str, 'int': Int, 'float': Float}


def is_json_encoded(encoding: str, value: Any) -> bool:
    """Get whether the given object is of this encoding type.

    Args:
        encoding (str): The encoding.
        value (Any): The object.

    Returns:
        bool: Whether of this type.
    """
    cls = _encodings[encoding]
    return cls.is_encoded(value)


def is_json_encoding(encoding: str) -> bool:
    """Get whether this is a supported encoding.

    Args:
        encoding (str): Encoding.

    Returns:
        bool: Whether encoding is supported.
    """
    return encoding in _encodings
