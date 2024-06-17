# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Encode and Decode samples in a supported MDS format."""

import json
import pickle
from abc import ABC, abstractmethod
from decimal import Decimal
from io import BytesIO
from typing import Any, Optional, Set, Tuple

import numpy as np
from numpy import typing as npt
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from typing_extensions import Self

__all__ = [
    'get_mds_encoded_size', 'get_mds_encodings', 'is_mds_encoding', 'mds_decode', 'mds_encode',
    'is_mds_encoding_safe'
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


class NDArray(Encoding):
    """Store NumPy NDArray.

    The dtype and shape may be either static or dynamic.

    Accordingly, there are four serialized formats:
      * Static dtype:
          * Static shape:
              [values: size * dtype]
          * Dynamic shape:
              [ndim | shape dtype: 1] [shape: ndim * shape dtype] [values: size * dtype]
      * Dynamic dtype:
          * Static shape:
              [dtype: 1] [values: size * dtype]
          * Dynamic shape:
              [dtype: 1] [ndim | shape dtype: 1] [shape: ndim * shape dtype] [values: size * dtype]

    Args:
        dtype (str, optional): The dtype, if fixed. Defaults to ``None``.
        shape (Tuple[int], optional): The shape, if fixed. Defaults to ``None``.
    """

    # Integer <4 -> shape dtype.
    _int2shape_dtype = {
        0: 'uint8',
        1: 'uint16',
        2: 'uint32',
        3: 'uint64',
    }

    # Shape dtype -> integer <4.
    _shape_dtype2int = {v: k for k, v in _int2shape_dtype.items()}

    # Integer <256 -> value dtype.
    _int2value_dtype = {
        8: 'uint8',
        9: 'int8',
        16: 'uint16',
        17: 'int16',
        18: 'float16',
        32: 'uint32',
        33: 'int32',
        34: 'float32',
        64: 'uint64',
        65: 'int64',
        66: 'float64',
    }

    # Value dtype -> integer <256.
    _value_dtype2int = {v: k for k, v in _int2value_dtype.items()}

    @classmethod
    def _get_static_size(cls, dtype: Optional[str], shape: Optional[Tuple[int]]) -> Optional[int]:
        """Get the fixed size of the column in bytes, if applicable.

        Args:
            dtype (str, optional): The dtype, if fixed.
            shape (Tuple[int], optional): The shape, if fixed.

        Returns:
            int: The fixed size in bytes, if there is one.
        """
        if dtype is None or shape is None:
            return None
        return int(np.prod(shape)) * getattr(np, dtype)().nbytes

    def __init__(self, dtype: Optional[str] = None, shape: Optional[Tuple[int]] = None):
        if dtype is not None:
            assert dtype in self._value_dtype2int
        if shape is not None:
            for dim in shape:
                assert 1 <= dim
        self.dtype = dtype
        self.shape = shape
        self.size = self._get_static_size(dtype, shape)

    @classmethod
    def from_str(cls, text: str) -> Self:
        """Parse this encoding from string.

        Args:
            text (str): The string to parse.

        Returns:
            Self: The initialized Encoding.
        """
        args = text.split(':') if text else []
        assert len(args) in {0, 1, 2}
        if 1 <= len(args):
            dtype = args[0]
        else:
            dtype = None
        if 2 <= len(args):
            shape = tuple(map(int, args[1].split(',')))
        else:
            shape = None
        return cls(dtype, shape)

    @classmethod
    def _rightsize_shape_dtype(cls, shape: npt.NDArray[np.int64]) -> str:
        """Get the smallest unsigned int dtype that will accept the given shape.

        Args:
            shape (NDArray[np.int64]): The shape.

        Returns:
            str: The smallest acceptable uint* dtype.
        """
        if len(shape) == 0:
            raise ValueError(
                'Attempting to encode a scalar with NDArray encoding. Please use a scalar encoding.'
            )

        if shape.min() <= 0:
            raise ValueError('All dimensions must be greater than zero.')
        x = shape.max()
        if x < (1 << 8):
            return 'uint8'
        elif x < (1 << 16):
            return 'uint16'
        elif x < (1 << 32):
            return 'uint32'
        else:
            return 'uint64'

    def encode(self, obj: npt.NDArray) -> bytes:
        """Encode the given data from the original object to bytes.

        Args:
            obj (NDArray): Decoded object.

        Returns:
            bytes: Encoded data.
        """
        parts = []

        # Encode dtype, if not given in header.
        dtype_int = self._value_dtype2int.get(obj.dtype.name)
        if dtype_int is None:
            raise ValueError(f'Unsupported dtype: {obj.dtype.name}.')
        if self.dtype is None:
            part = bytes([dtype_int])
            parts.append(part)
        else:
            if obj.dtype != self.dtype:
                raise ValueError(f'Wrong dtype: expected {self.dtype}, got {obj.dtype.name}.')

        if obj.size == 0:
            raise ValueError('Attempting to encode a numpy array with 0 elements.')

        # Encode shape, if not given in header.
        if self.shape is None:
            ndim = len(obj.shape)
            if 64 <= ndim:
                raise ValueError('Array has too many axes: maximum 63, got {ndim}.')
            shape_arr = np.array(obj.shape, np.int64)
            shape_dtype = self._rightsize_shape_dtype(shape_arr)
            shape_dtype_int = self._shape_dtype2int[shape_dtype]
            byte = (ndim << 2) | shape_dtype_int
            part = bytes([byte])
            parts.append(part)
            part = shape_arr.astype(shape_dtype).tobytes()
            parts.append(part)
        else:
            if obj.shape != self.shape:
                raise ValueError('Wrong shape: expected {self.shape}, got {obj.shape}.')

        # Encode the array values.
        part = obj.tobytes()
        parts.append(part)

        return b''.join(parts)

    def decode(self, data: bytes) -> npt.NDArray:
        """Decode the given data from bytes to the original object.

        Args:
            data (bytes): Encoded data.

        Returns:
            NDArray: Decoded object.
        """
        index = 0

        # Decode dtype, if not given in header.
        if self.dtype:
            dtype = self.dtype
        else:
            dtype_int = data[index]
            index += 1
            dtype = self._int2value_dtype[dtype_int]

        # Decode shape, if not given in header.
        if self.shape:
            shape = self.shape
        else:
            byte = data[index]
            index += 1
            ndim = byte >> 2
            shape_dtype_int = byte % 4
            shape_dtype = self._int2shape_dtype[shape_dtype_int]
            shape_dtype_nbytes = 2**shape_dtype_int
            size = ndim * shape_dtype_nbytes
            shape = np.frombuffer(data[index:index + size], shape_dtype)
            index += size

        # Decode the array values.
        arr = np.frombuffer(data[index:], dtype)
        return arr.reshape(shape)  # pyright: ignore


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


class StrEncoding(Encoding):
    """Base class for stringified types.

    Using variable-length strings allows us to store scalars with arbitrary precision.

    The encode/decode methods of subclasses are the same except for typing specializations.
    """

    pass


class StrInt(StrEncoding):
    """Store int as variable-length digits str."""

    def encode(self, obj: int) -> bytes:
        self._validate(obj, int)
        return str(obj).encode('utf-8')

    def decode(self, data: bytes) -> int:
        return int(data.decode('utf-8'))


class StrFloat(Encoding):
    """Store float as variable-length digits str."""

    def encode(self, obj: float) -> bytes:
        self._validate(obj, float)
        return str(obj).encode('utf-8')

    def decode(self, data: bytes) -> float:
        return float(data.decode('utf-8'))


class StrDecimal(Encoding):
    """Store decimal as variable-length digits str."""

    def encode(self, obj: Decimal) -> bytes:
        self._validate(obj, Decimal)
        return str(obj).encode('utf-8')

    def decode(self, data: bytes) -> Decimal:
        return Decimal(data.decode('utf-8'))


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
    'ndarray': NDArray,
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
    'str_int': StrInt,
    'str_float': StrFloat,
    'str_decimal': StrDecimal,
    'pil': PIL,
    'jpeg': JPEG,
    'png': PNG,
    'pkl': Pickle,
    'json': JSON,
}

_unsafe_encodings = {'pkl'}


def get_mds_encodings() -> Set[str]:
    """List supported encodings.

    Returns:
        Set[str]: Encoding names.
    """
    return set(_encodings)


def _get_coder(encoding: str) -> Optional[Encoding]:
    """Get an object that encodes/decodes.

    Args:
        encoding (str): The encoding details.

    Returns:
        Encoding: The coder.
    """
    index = encoding.find(':')
    if index == -1:
        cls = _encodings.get(encoding)
        if cls is None:
            return None
        return cls()
    name = encoding[:index]
    config = encoding[index + 1:]
    return _encodings[name].from_str(config)


def is_mds_encoding(encoding: str) -> bool:
    """Get whether the given encoding is supported.

    Args:
        encoding (str): Encoding.

    Returns:
        bool: Whether the encoding is valid.
    """
    coder = _get_coder(encoding)
    return coder is not None


def is_mds_encoding_safe(encoding: str) -> bool:
    """Get whether the given encoding is safe (does not allow arbitrary code execution).

    Args:
        encoding (str): Encoding.

    Returns:
        bool: Whether the encoding is safe.
    """
    return encoding not in _unsafe_encodings


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
    coder = _get_coder(encoding)
    if coder is None:
        raise ValueError(f'Unsupported encoding: {encoding}.')
    return coder.encode(obj)


def mds_decode(encoding: str, data: bytes) -> Any:
    """Decode the given data from bytes to the original object.

    Args:
        encoding (str): Encoding.
        data (bytes): Encoded data.

    Returns:
        Any: Decoded object.
    """
    coder = _get_coder(encoding)
    if coder is None:
        raise ValueError(f'Unsupported encoding: {encoding}.')
    return coder.decode(data)


def get_mds_encoded_size(encoding: str) -> Optional[int]:
    """Get the fixed size of all encodings of this type, or None if N/A.

    Args:
        encoding (str): Encoding.

    Returns:
        Optional[int]: Size of encoded data.
    """
    coder = _get_coder(encoding)
    if coder is None:
        raise ValueError(f'Unsupported encoding: {encoding}.')
    return coder.size
