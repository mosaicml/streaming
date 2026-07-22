# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Dict DBS type."""

from typing import Dict as TyDict
from typing import Tuple as TyTuple

import numpy as np

from streaming.base.format.dbs.type.base import Tree, decode_int, get_template, prepend_int


class Dict(Tree):
    """Dict DBS type."""

    py_type = dict

    def encode(self, obj: TyDict) -> bytes:
        from streaming.base.format.dbs.type import get_encoder
        keys, vals = zip(*obj.items())
        key_py_type = get_template(keys)
        val_py_type = get_template(vals)
        key_dbs_type_id, key_coder = get_encoder(key_py_type)
        val_dbs_type_id, val_coder = get_encoder(val_py_type)
        ret = [key_dbs_type_id.tobytes(), val_dbs_type_id.tobytes()]
        for key in sorted(obj):
            val = obj[key]
            key_data = key_coder.encode(key)
            val_data = val_coder.encode(val)
            ret += [key_data, val_data]
        data = b''.join(ret)
        return prepend_int(np.uint32, len(obj), data)

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TyDict, int]:
        from streaming.base.format.dbs.type import get_decoder
        count, offset = decode_int(data, offset, np.uint32)
        key_dbs_type_id = np.uint8(data[offset])
        val_dbs_type_id = np.uint8(data[offset + 1])
        offset += key_dbs_type_id.nbytes + val_dbs_type_id.nbytes
        key_coder = get_decoder(key_dbs_type_id)
        val_coder = get_decoder(val_dbs_type_id)
        obj = {}
        for _ in range(count):
            key, offset = key_coder.decode(data, offset)
            val, offset = val_coder.decode(data, offset)
            if key in obj:
                raise ValueError(f'Duplicate key found in the serialization of a DBS dict ' +
                                 f'type: {key}.')
            obj[key] = val
        return obj, offset
