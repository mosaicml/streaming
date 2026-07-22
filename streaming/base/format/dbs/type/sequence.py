# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Sequence DBS types."""

from typing import Sequence as TySequence
from typing import Tuple as TyTuple

import numpy as np

from streaming.base.format.dbs.type.base import Tree, decode_int, get_template, prepend_int


class Sequence(Tree):
    """Sequence DBS type abstract base class."""

    def encode(self, obj: TySequence) -> bytes:
        from streaming.base.format.dbs.type import get_encoder
        py_type = get_template(obj)
        dbs_type_id, coder = get_encoder(py_type)
        ret = [dbs_type_id.tobytes()]
        for item in obj:
            item_data = coder.encode(item)
            ret.append(item_data)
        data = b''.join(ret)
        return prepend_int(np.uint32, len(obj), data)

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TySequence, int]:
        from streaming.base.format.dbs.type import get_decoder
        count, offset = decode_int(data, offset, np.uint32)
        dbs_type_id = np.uint8(data[offset])
        offset += dbs_type_id.nbytes
        coder = get_decoder(dbs_type_id)
        items = []
        for _ in range(count):
            item, offset = coder.decode(data, offset)
            items.append(item)
        seq = self.py_type(items)
        return seq, offset


class List(Sequence):
    """List DBS type."""

    py_type = list


class Tuple(Sequence):
    """Tuple DBS type."""

    py_type = tuple
