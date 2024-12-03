# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""JSON DBS type."""

import json
from typing import Any as TyAny
from typing import Tuple as TyTuple

import numpy as np

from streaming.base.format.dbs.type.base import ComplexVarLeaf, decode_int, prepend_int


class JSON(ComplexVarLeaf):
    """JSON DBS type.

    Notes:
    - Human-readable format.
    - But only supports dict, list, str, int, float.
    """

    def encode(self, obj: TyAny) -> bytes:
        text = json.dumps(obj, sort_keys=True)
        data = text.encode('utf-8')
        return prepend_int(np.uint32, len(data), data)

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TyAny, int]:
        size, offset = decode_int(data, offset, np.uint32)
        buf = data[offset:offset + size]
        text = buf.decode('utf-8')
        obj = json.loads(text)
        return obj, offset + size
