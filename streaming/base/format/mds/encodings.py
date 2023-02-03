# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Encode and Decode samples in a supported MDS format."""

import json
import pickle
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Optional, Set

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile

__all__ = [
    'get_mds_encoded_size', 'get_mds_encodings', 'is_mds_encoding', 'mds_decode', 'mds_encode'
]


class Encoding(ABC):
    """Encodes and decodes between objects of a certain type and raw bytes."""

    size: Optional[int] = None  # Fixed size in bytes of encoded data (None if variable size).

    @abstractmethod
    def encode(self, obj: Any) -> bytes:
        """Encode the given data from the original object to bytes.

        Args:
            obj (Any): Decoded object.

        Returns:
            bytes: Encoded data.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, data: bytes) -> Any:
        """Decode the given data from bytes to the original object.

        Args:
            data (bytes): Encoded data.

        Returns:
            Any: Decoded object.
        """
        raise NotImplementedError

    @staticmethod
    def _validate(data: Any, expected_type: Any) -> None:
        if not isinstance(data, expected_type):
            raise AttributeError(
                f'data should be of type {expected_type}, but instead, found as {type(data)}')


class Bytes(Encoding):
    """Store bytes (no-op encoding)."""

    def encode(self, obj: bytes) -> bytes:
        self._validate(obj, bytes)
        return obj

    def decode(self, data: bytes) -> bytes:
        return data


class Str(Encoding):
    """Store UTF-8."""

    def encode(self, obj: str) -> bytes:
        self._validate(obj, str)
        return obj.encode('utf-8')

    def decode(self, data: bytes) -> str:
        return data.decode('utf-8')


class Int(Encoding):
    """Store int64."""

    size = 8

    def encode(self, obj: int) -> bytes:
        self._validate(obj, int)
        return np.int64(obj).tobytes()

    def decode(self, data: bytes) -> int:
        return int(np.frombuffer(data, np.int64)[0])


class Scalar(Encoding):
    """Store scalar."""

    def __init__(self, dtype: type) -> None:
        self.dtype = dtype
        self.size = self.dtype().nbytes

    def encode(self, obj: Any) -> bytes:
        return self.dtype(obj).tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, self.dtype)[0]


class UInt8(Scalar):
    """Store uint8."""

    def __init__(self):
        super().__init__(np.uint8)


class UInt16(Scalar):
    """Store uint16."""

    def __init__(self):
        super().__init__(np.uint16)


class UInt32(Scalar):
    """Store uint32."""

    def __init__(self):
        super().__init__(np.uint32)


class UInt64(Scalar):
    """Store uint64."""

    def __init__(self):
        super().__init__(np.uint64)


class Int8(Scalar):
    """Store int8."""

    def __init__(self):
        super().__init__(np.int8)


class Int16(Scalar):
    """Store int16."""

    def __init__(self):
        super().__init__(np.int16)


class Int32(Scalar):
    """Store int32."""

    def __init__(self):
        super().__init__(np.int32)


class Int64(Scalar):
    """Store int64."""

    def __init__(self):
        super().__init__(np.int64)


class Float16(Scalar):
    """Store float16."""

    def __init__(self):
        super().__init__(np.float16)


class Float32(Scalar):
    """Store float32."""

    def __init__(self):
        super().__init__(np.float32)


class Float64(Scalar):
    """Store float64."""

    def __init__(self):
        super().__init__(np.float64)


class PIL(Encoding):
    """Store PIL image raw.

    Format: [width: 4] [height: 4] [mode size: 4] [mode] [raw image].
    """

    def encode(self, obj: Image.Image) -> bytes:
        self._validate(obj, Image.Image)
        mode = obj.mode.encode('utf-8')
        width, height = obj.size
        raw = obj.tobytes()
        ints = np.array([width, height, len(mode)], np.uint32)
        return ints.tobytes() + mode + raw

    def decode(self, data: bytes) -> Image.Image:
        idx = 3 * 4
        width, height, mode_size = np.frombuffer(data[:idx], np.uint32)
        idx2 = idx + mode_size
        mode = data[idx:idx2].decode('utf-8')
        size = width, height
        raw = data[idx2:]
        return Image.frombytes(mode, size, raw)  # pyright: ignore


class JPEG(Encoding):
    """Store PIL image as JPEG."""

    def encode(self, obj: Image.Image) -> bytes:
        self._validate(obj, Image.Image)
        if isinstance(obj, JpegImageFile) and hasattr(obj, 'filename'):
            # read the source file to prevent lossy re-encoding
            with open(obj.filename, 'rb') as f:
                return f.read()
        else:
            out = BytesIO()
            obj.save(out, format='JPEG')
            return out.getvalue()

    def decode(self, data: bytes) -> Image.Image:
        inp = BytesIO(data)
        return Image.open(inp)


class PNG(Encoding):
    """Store PIL image as PNG."""

    def encode(self, obj: Image.Image) -> bytes:
        self._validate(obj, Image.Image)
        out = BytesIO()
        obj.save(out, format='PNG')
        return out.getvalue()

    def decode(self, data: bytes) -> Image.Image:
        inp = BytesIO(data)
        return Image.open(inp)


class Pickle(Encoding):
    """Store arbitrary data as pickle."""

    def encode(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def decode(self, data: bytes) -> Any:
        return pickle.loads(data)


class JSON(Encoding):
    """Store arbitrary data as JSON."""

    def encode(self, obj: Any) -> bytes:
        data = json.dumps(obj)
        self._is_valid(obj, data)
        return data.encode('utf-8')

    def decode(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))

    def _is_valid(self, original: Any, converted: Any) -> None:
        try:
            json.loads(converted)
        except json.decoder.JSONDecodeError as e:
            e.msg = f'Invalid JSON data: {original}'
            raise


# Encodings (name -> class).
_encodings = {
    'bytes': Bytes,
    'str': Str,
    'int': Int,
    'uint8': UInt8,
    'uint16': UInt16,
    'uint32': UInt32,
    'uint64': UInt64,
    'int8': Int8,
    'int16': Int16,
    'int32': Int32,
    'int64': Int64,
    'float16': Float16,
    'float32': Float32,
    'float64': Float64,
    'pil': PIL,
    'jpeg': JPEG,
    'png': PNG,
    'pkl': Pickle,
    'json': JSON,
}


def get_mds_encodings() -> Set[str]:
    """List supported encodings.

    Returns:
        Set[str]: Encoding names.
    """
    return set(_encodings)


def is_mds_encoding(encoding: str) -> bool:
    """Get whether the given encoding is supported.

    Args:
        encoding (str): Encoding.

    Returns:
        bool: Whether the encoding is valid.
    """
    return encoding in _encodings


def mds_encode(encoding: str, obj: Any) -> bytes:
    """Encode the given data from the original object to bytes.

    Args:
        encoding (str): Encoding.
        obj (Any): Decoded object.

    Returns:
        bytes: Encoded data.
    """
    if isinstance(obj, bytes):
        return obj
    cls = _encodings[encoding]
    return cls().encode(obj)


def mds_decode(encoding: str, data: bytes) -> Any:
    """Decode the given data from bytes to the original object.

    Args:
        encoding (str): Encoding.
        data (bytes): Encoded data.

    Returns:
        Any: Decoded object.
    """
    cls = _encodings[encoding]
    return cls().decode(data)


def get_mds_encoded_size(encoding: str) -> Optional[int]:
    """Get the fixed size of all encodings of this type, or None if N/A.

    Args:
        encoding (str): Encoding.

    Returns:
        Optional[int]: Size of encoded data.
    """
    cls = _encodings[encoding]
    return cls().size
