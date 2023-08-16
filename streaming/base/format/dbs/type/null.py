# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Null DBS type."""

from typing import Tuple as TyTuple

from streaming.base.format.dbs.type.base import FixLeaf


class Null(FixLeaf):
    """Null DBS type."""

    py_type = type(None)

    def encode(self, obj: None) -> bytes:
        return b''

    encoded_size = 0

    def decode(self, data: bytes, offset: int = 0) -> TyTuple[None, int]:
        return None, offset
