# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Store complex types in MDS."""

import json
import pickle
from typing import Any

from streaming.base.format.mds.encodings.base import Encoding

__all__ = ['JSON', 'Pickle']


class JSON(Encoding):
    """Store arbitrary data as JSON."""

    def encode(self, obj: Any) -> bytes:
        data = json.dumps(obj)
        self._is_valid(obj, data)
        return data.encode('utf-8')

    def decode(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))

    def _is_valid(self, original: Any, converted: Any) -> None:
        try:
            json.loads(converted)
        except json.decoder.JSONDecodeError as e:
            e.msg = f'Invalid JSON data: {original}'
            raise


class Pickle(Encoding):
    """Store arbitrary data as pickle."""

    def encode(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def decode(self, data: bytes) -> Any:
        return pickle.loads(data)
