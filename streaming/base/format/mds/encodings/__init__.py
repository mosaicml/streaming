# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Store data in MDS."""

from typing import Any, Optional, Set

from streaming.base.format.mds.encodings.base import Encoding
from streaming.base.format.mds.encodings.complex import JSON, Pickle
from streaming.base.format.mds.encodings.image import JPEG, PIL, PNG
from streaming.base.format.mds.encodings.scalar import (Float16, Float32, Float64, Int, Int8,
                                                        Int16, Int32, Int64, UInt8, UInt16, UInt32,
                                                        UInt64)
from streaming.base.format.mds.encodings.sequence import Bytes, Str
from streaming.base.format.mds.encodings.tensor import NDArray

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
