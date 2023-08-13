# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Store sequences in MDS."""

from streaming.base.format.mds.encodings.base import Encoding

__all__ = ['Bytes', 'Str']


class Bytes(Encoding):
    """Store bytes."""

    def encode(self, obj: bytes) -> bytes:
        self._validate(obj, bytes)
        return obj

    def decode(self, data: bytes) -> bytes:
        return data


class Str(Encoding):
    """Store UTF-8."""

    def encode(self, obj: str) -> bytes:
        self._validate(obj, str)
        return obj.encode('utf-8')

    def decode(self, data: bytes) -> str:
        return data.decode('utf-8')
