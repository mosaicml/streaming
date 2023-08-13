# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Store ndarrays in MDS."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray as NumpyNDArray
from typing_extensions import Self

from streaming.base.format.mds.encodings.base import Encoding

__all__ = ['NDArray']


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
    def _rightsize_shape_dtype(cls, shape: NumpyNDArray[np.int64]) -> str:
        """Get the smallest unsigned int dtype that will accept the given shape.

        Args:
            shape (NumpyNDArray[np.int64]): The shape.

        Returns:
            str: The smallest acceptable uint* dtype.
        """
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

    def encode(self, obj: NumpyNDArray) -> bytes:
        """Encode the given data from the original object to bytes.

        Args:
            obj (NumpyNDArray): Decoded object.

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
                raise ValueError('Wrong dtype: expected {self.dtype}, got {obj.dtype.name}.')

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

    def decode(self, data: bytes) -> NumpyNDArray:
        """Decode the given data from bytes to the original object.

        Args:
            data (bytes): Encoded data.

        Returns:
            NumpyNDArray: Decoded object.
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
