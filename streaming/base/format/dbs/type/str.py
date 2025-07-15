# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Str DBS type."""

from typing import Tuple as TyTuple

import numpy as np

from streaming.base.format.dbs.type.base import SimpleVarLeaf, decode_int, prepend_int


class Str(SimpleVarLeaf):
    """Str DBS type."""

    py_type = str

    def encode(self, obj: str) -> bytes:
        obj_data = obj.encode('utf-8')
        return prepend_int(np.uint32, len(obj_data), obj_data)

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[str, int]:
        size, offset = decode_int(data, offset, np.uint32)
        obj_data = data[offset:offset + size]
        obj = obj_data.decode('utf-8')
        return obj, offset + size
