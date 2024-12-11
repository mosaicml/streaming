# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Any DBS type."""

from typing import Any as TyAny
from typing import Optional as TyOptional
from typing import Tuple as TyTuple

import numpy as np
from numpy.typing import NDArray as TyNDArray

from streaming.base.format.dbs.type.base import DBSType


class Union(DBSType):
    """Union DBS type.

    Notes:
    - Accepts types according to the type acceptance array specified in init.
    - If it is None, accepts all types.
    """

    def __init__(self, contains: TyOptional[TyNDArray[np.uint8]]) -> None:
        self.contains = contains

    py_type: None = None  # Corresponds to multiple possible deserialized python types.

    def encode(self, obj: TyAny) -> bytes:
        from streaming.base.format.dbs.type import get_encoder
        py_type = type(obj)
        dbs_type_id, coder = get_encoder(py_type)
        if self.contains is not None and not self.contains[dbs_type_id]:
            raise ValueError(f'The given type is not in this union: {type(coder)}.')
        data = coder.encode(obj)
        return dbs_type_id.tobytes() + data

    encoded_size: None = None  # Serialized size varies.

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[TyAny, int]:
        from streaming.base.format.dbs.type import get_decoder
        dbs_type_id = np.uint8(data[offset])
        coder = get_decoder(dbs_type_id)
        if self.contains is not None and not self.contains[dbs_type_id]:
            raise ValueError(f'The given type is not in this union: {type(coder)}.')
        return coder.decode(data, offset + 1)


class Any(Union):
    """Any DBS type.

    Notes:
    - Accepts every type.
    """

    def __init__(self) -> None:
        super().__init__(None)


class Just(Union):
    """Just DBS type.

    Notes:
    - Accepts only the type specified in init.
    """

    def __init__(self, dbs_type_id: np.uint8) -> None:
        contains = np.zeros(256, np.uint8)
        contains[dbs_type_id] = 1
        super().__init__(contains)


class Maybe(Union):
    """Maybe DBS type.

    Notes:
    - Accepts only the type specified in init and None.
    """

    def __init__(self, dbs_type_id: np.uint8) -> None:
        from streaming.base.format.dbs.type import get_null_dbs_type_id
        contains = np.zeros(256, np.uint8)
        contains[dbs_type_id] = 1
        null_dbs_type_id = get_null_dbs_type_id()
        contains[null_dbs_type_id] = 1
        super().__init__(contains)
