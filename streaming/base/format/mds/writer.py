# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional

import numpy as np

from streaming.base.format.base.writer import JointWriter
from streaming.base.format.mds.encodings import get_mds_encoded_size, is_mds_encoding, mds_encode


class MDSWriter(JointWriter):
    format = 'mds'
    extra_bytes_per_sample = 4

    def __init__(self,
                 dirname: str,
                 columns: Dict[str, str],
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26) -> None:
        super().__init__(dirname, compression, hashes, size_limit, 0, self.extra_bytes_per_sample)

        self.columns = columns
        self.column_names = []
        self.column_encodings = []
        self.column_sizes = []
        for name in sorted(columns):
            encoding = columns[name]
            assert is_mds_encoding(encoding)
            size = get_mds_encoded_size(encoding)
            self.column_names.append(name)
            self.column_encodings.append(encoding)
            self.column_sizes.append(size)

        obj = self.get_config()
        text = json.dumps(obj, sort_keys=True)
        self.config_data = text.encode('utf-8')
        self.extra_bytes_per_shard = 4 + 4 + len(self.config_data)
        self._reset_cache()

    def encode_sample(self, sample: Dict[str, Any]) -> bytes:
        sizes = []
        data = []
        for key, encoding, size in zip(self.column_names, self.column_encodings, self.column_sizes):
            value = sample[key]
            datum = mds_encode(encoding, value)
            if size is None:
                size = len(datum)
                sizes.append(size)
            else:
                assert size == len(datum)
            data.append(datum)
        head = np.array(sizes, np.uint32).tobytes()
        body = b''.join(data)
        return head + body

    def get_config(self) -> Dict[str, Any]:
        obj = super().get_config()
        obj.update({
            'column_names': self.column_names,
            'column_encodings': self.column_encodings,
            'column_sizes': self.column_sizes
        })
        return obj

    def encode_joint_shard(self) -> bytes:
        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        offsets += len(num_samples.tobytes()) + len(offsets.tobytes()) + len(self.config_data)
        sample_data = b''.join(self.new_samples)
        return num_samples.tobytes() + offsets.tobytes() + self.config_data + sample_data
