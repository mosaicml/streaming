import json
from typing import Any, Optional

import numpy as np

from ..base.writer import SplitWriter
from .encodings import is_json_encoded, is_json_encoding


class JSONWriter(SplitWriter):
    format = 'json'

    def __init__(self,
                 dirname: str,
                 columns: dict[str, str],
                 compression: Optional[str] = None,
                 hashes: Optional[list[str]] = None,
                 size_limit: Optional[int] = 1 << 26,
                 newline: str = '\n') -> None:
        super().__init__(dirname, compression, hashes, size_limit)

        for encoding in columns.values():
            assert is_json_encoding(encoding)

        self.columns = columns
        self.newline = newline

    def encode_sample(self, sample: dict[str, Any]) -> bytes:
        obj = {}
        for key, encoding in self.columns.items():
            value = sample[key]
            assert is_json_encoded(encoding, value)
            obj[key] = value
        text = json.dumps(obj, sort_keys=True) + self.newline
        return text.encode('utf-8')

    def get_config(self) -> dict[str, Any]:
        obj = super().get_config()
        obj.update({'columns': self.columns, 'newline': self.newline})
        return obj

    def encode_split_shard(self) -> tuple[bytes, bytes]:
        data = b''.join(self.new_samples)

        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        obj = self.get_config()
        text = json.dumps(obj, sort_keys=True)
        meta = num_samples.tobytes() + offsets.tobytes() + text.encode('utf-8')

        return data, meta
