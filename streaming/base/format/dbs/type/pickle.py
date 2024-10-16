# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Pickle DBS type."""

import pickle
from typing import Any as TyAny
from typing import Tuple as TyTuple

import numpy as np

from streaming.base.format.dbs.type.base import ComplexVarLeaf, decode_int, prepend_int


class Pickle(ComplexVarLeaf):
    """Pickle DBS type.

    Notes:
    - Binary format.
    - Supports almost any type you can imagine.
    - Potentially very inefficient serialization format.
    - Images are serialized as their raw data tensor, uncopressed.
    - When you are unfortunate enough to need it, it is there.
    """

    def encode(self, obj: TyAny) -> bytes:
        data = pickle.dumps(obj)
        return prepend_int(np.uint32, len(data), data)

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TyAny, int]:
        size, offset = decode_int(data, offset, np.uint32)
        buf = data[offset:offset + size]
        obj = pickle.loads(buf)
        return obj, offset + size
