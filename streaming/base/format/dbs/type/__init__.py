# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""DBS types."""

from typing import Dict as TyDict
from typing import List as TyList
from typing import Optional as TyOptional
from typing import Tuple as TyTuple

import numpy as np

from streaming.base.format.dbs.type.base import DBSType
from streaming.base.format.dbs.type.bytes import Bytes
from streaming.base.format.dbs.type.dict import Dict
from streaming.base.format.dbs.type.image import JPG, PNG, RawImage
from streaming.base.format.dbs.type.json import JSON
from streaming.base.format.dbs.type.ndarray import NDArray
from streaming.base.format.dbs.type.null import Null
from streaming.base.format.dbs.type.number import (Bool, Float, Float16, Float32, Float64, Int,
                                                   Int8, Int16, Int32, Int64, UInt8, UInt16,
                                                   UInt32, UInt64)
from streaming.base.format.dbs.type.pickle import Pickle
from streaming.base.format.dbs.type.sequence import List, Tuple
from streaming.base.format.dbs.type.str import Str
from streaming.base.format.dbs.type.union import Any, Just, Maybe, Union

# List of all DBS types.
#
# Notes:
# - Hard limit of 256 DBS types (which type something is is stored in a byte).
# - DBS type ID is determined from its index in this list.
# - It is fine to create new DBS types and append them to the end of this list.
# - It is pointless but safe to list a DBS type multiple times in this list.
# - You must NEVER change or remove DBS types once they are in this list!
_dbs_types: TyList[type] = [
    Null,
    Union,
    Any,
    Just,
    Maybe,
    Bool,
    Int,
    Float,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    NDArray,
    Bytes,
    Str,
    RawImage,
    JPG,
    PNG,
    Dict,
    List,
    Tuple,
    JSON,
    Pickle,
]

# Mapping of native type to DBS type ID.
_py_type2dbs_type_id: TyDict[type, np.uint8] = {}
for dbs_type_id, dbs_type in enumerate(_dbs_types):
    assert dbs_type not in _py_type2dbs_type_id
    _py_type2dbs_type_id[dbs_type().py_type] = np.uint8(dbs_type_id)

# Special DBS type IDs.
_null_dbs_type_id = np.uint8(_dbs_types.index(Null))
_any_dbs_type_id = np.uint8(_dbs_types.index(Any))


def get_null_dbs_type_id() -> np.uint8:
    """Get the ID of the Null DBS type.

    Returns:
        np.uint8: DBS type ID of Null.
    """
    return _null_dbs_type_id


def get_encoder(py_type: TyOptional[type]) -> TyTuple[np.uint8, DBSType]:
    """Get a DBS type by its corresponding python type.

    Args:
        py_type (TyOptional[type]): Python type when decoded, or None for Any.

    Returns:
        TyTuple[int, DBSType]: DBS type ID and default instance of that DBS type.
    """
    if not py_type:
        return _any_dbs_type_id, Any()
    dbs_type_id = _py_type2dbs_type_id.get(py_type)
    if dbs_type_id is None:
        raise ValueError('No DBS type for python type: {py_type}.')
    dbs_type = _dbs_types[dbs_type_id]
    return dbs_type_id, dbs_type()


def get_decoder(dbs_type_id: np.uint8) -> DBSType:
    """Get a DBS type by DBS type ID.

    Args:
        dbs_type_id (np.uint8): DBS type ID.

    Returns:
        DBSType: Default instance of that DBS type.
    """
    dbs_type = _dbs_types[dbs_type_id]
    return dbs_type()
