# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Bytes DBS type."""

from typing import Tuple as TyTuple

import numpy as np

from streaming.base.format.dbs.type.base import SimpleVarLeaf, decode_int, prepend_int


class Bytes(SimpleVarLeaf):
    """Bytes DBS type."""

    py_type = bytes

    def encode(self, obj: bytes) -> bytes:
        return prepend_int(np.uint32, len(obj), obj)

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[bytes, int]:
        size, offset = decode_int(data, offset, np.uint32)
        obj = data[offset:offset + size]
        return obj, offset + size
