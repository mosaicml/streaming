# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Serialize data in DBS."""

from abc import ABC, abstractmethod
from typing import Any as TyAny
from typing import Optional as TyOptional
from typing import Sequence as TySequence
from typing import Tuple as TyTuple
from typing import Union as TyUnion

import numpy as np


def get_template(items: TySequence) -> TyOptional[type]:
    """Get the type in common of the given items.

    Args:
        items (TySequence): The given items.

    Returns:
        TyOptional[type]: Their type in common, if any.
    """
    py_types = list(map(type, items))
    if len(set(py_types)) == 1:
        return py_types[0]
    else:
        return None


def prepend_int(dtype: type, num: TyUnion[int, np.integer], data: bytes) -> bytes:
    """Prepend the given data with a numpy int.

    Args:
        dtype (type): Original data.
        num (TyUnion[int, np.integer]): Int to prepend.
        data (bytes): Serialized int dtype.

    Returns:
        bytes: Prepended data.
    """
    return dtype(num).tobytes() + data


def prepend_float(dtype: type, num: TyUnion[float, np.floating], data: bytes) -> bytes:
    """Prepend the given data with a numpy float.

    Args:
        dtype (type): Original data.
        num (TyUnion[float, np.floating]): Float to prepend.
        data (bytes): Serialized float dtype.

    Returns:
        bytes: Prepended data.
    """
    return dtype(num).tobytes() + data


def decode_int(data: bytes, offset: int, dtype: type) -> TyTuple[np.integer, int]:
    """Decode a prepended numpy int from the given data.

    Args;
        data (bytes): Prepended data.
        offset (int): Offset into the data.
        dtype (type): Serialized int dtype.

    Returns:
        TyTuple[np.integer, int]: Prepended int and new offset.
    """
    num_size = dtype().nbytes
    num_data = data[offset:offset + num_size]
    num, = np.frombuffer(num_data, dtype)
    return num, offset + num_size


def decode_float(data: bytes, offset: int, dtype: type) -> TyTuple[np.floating, int]:
    """Decode a prepended numpy float from the given data.

    Args;
        data (bytes): Prepended data.
        offset (int): Offset into the data.
        dtype (type): Serialized float dtype.

    Returns:
        TyTuple[np.floating, int]: Prepended float and new offset.
    """
    num_size = dtype().nbytes
    num_data = data[offset:offset + num_size]
    num, = np.frombuffer(num_data, dtype)
    return num, offset + num_size


class DBSType(ABC):
    """DBS type abstract base class."""

    # Our corresponding python type, unless it is of dynamic type.
    py_type: TyOptional[type] = None

    @abstractmethod
    def encode(self, obj: TyAny) -> bytes:
        """Serialize python type to DBS data.

        Args:
            obj (TyAny): Python type.

        Returns:
            bytes: Serialized DBS data.
        """
        raise NotImplementedError

    # The fixed serialized size of this DBS type, unless it is of dynamic size.
    encoded_size: TyOptional[int] = None

    @abstractmethod
    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TyAny, int]:
        """Deserialize DBS data to python type.

        Args:
            data (bytes): Serialized DBS data.
            offset (int): Starting offset into data.

        Returns:
            TyTuple[TyAny, iny]: Pair of (deserialized python type, ending offset into data).
        """
        raise NotImplementedError


class Tree(DBSType):
    """Tree DBS type abstract base class.

    These types are recursive. They may contain any DBS type(s).

        ```
        Tree
        ├── Dict (dict.py)
        └── Sequence (sequence.py)
        ```
    """

    py_type: type  # One (recursive) DBS type maps to one (recursive) python type.
    encoded_size: None = None  # Serialized size varies.


class Leaf(DBSType):
    """Leaf (terminal) DBS type abstract base class.

    These types are terminal. They do not contain other types.

        ```
        Leaf
        ├── FixLeaf
        └── VarLeaf
        ```
    """

    pass  # No constraints over all leaf types.


class FixLeaf(Leaf):
    """Fixed-size leaf DBS type abstract base class.

    The serialized size of these types is known in advance.

        ```
        FixLeaf
        ├── Null (null.py)
        └── Number (number.py)
        ```
    """

    py_type: type  # One (fixed-size, terminal) DBS type maps to one (terminal) python type.
    encoded_size: int  # Has a specific fixed serialized size.


class VarLeaf(Leaf):
    """Variably-sized leaf DBS type abstract base class.

    The serialized size of these types must be prepended to the data.

        ```
        VarLeaf
        ├── ComplexVarLeaf
        └── SimpleVarLeaf
        ```
    """

    encoded_size: None = None  # Serialized size varies.


class SimpleVarLeaf(VarLeaf):
    """Simple variably-sized leaf DBS type abstract base class.

    These types consist of a single object.

        ```
        SimpleVarLeaf
        ├── Bytes (bytes.py)
        ├── Image (image.py)
        ├── NDArray (ndarray.py)
        └── Str (str.py)
        ```
    """

    py_type: type  # One (variably-sized, terminal) DBS type maps to one (terminal) python type.


class ComplexVarLeaf(VarLeaf):
    """Complex variably-sized leaf DBS type abstract base class.

    These types consist of structures of objects (without 'leaving" the type).

        ```
        ComplexVarLeaf
        ├── JSON (json.py)
        └── Pickle (pickle.py)
        ```
    """

    py_type: None = None  # Corresponds to multiple possible deserialized python types.
