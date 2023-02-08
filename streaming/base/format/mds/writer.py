# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

""":class:`MDSWriter` converts a list of samples into binary `.mds` files that can be read as a :class:`MDSReader`."""

import json
from typing import Any, Dict, List, Optional

import numpy as np

from streaming.base.format.base.writer import JointWriter
from streaming.base.format.mds.encodings import (get_mds_encoded_size, get_mds_encodings,
                                                 is_mds_encoding, mds_encode)

__all__ = ['MDSWriter']


class MDSWriter(JointWriter):
    """Writes a streaming MDS dataset.

    Args:
        dirname (str): Local dataset directory.
        columns (Dict[str, str]): Sample columns.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Defaults to ``1 << 26``.
    """

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
            if not is_mds_encoding(encoding):
                raise TypeError(
                    f'MDSWriter passed column "{name}" with encoding "{encoding}" is unsupported. Supported encodings are {get_mds_encodings()}'
                )
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
        """Encode a sample dict to bytes.

        Args:
            sample (Dict[str, Any]): Sample dict.

        Returns:
            bytes: Sample encoded as bytes.
        """
        sizes = []
        data = []
        for key, encoding, size in zip(self.column_names, self.column_encodings,
                                       self.column_sizes):
            value = sample[key]
            datum = mds_encode(encoding, value)
            if size is None:
                size = len(datum)
                sizes.append(size)
            else:
                if size != len(datum):
                    raise KeyError(
                        f'Unexpected data size; was this data typed with the correct encoding ({encoding})?'
                    )
            data.append(datum)
        head = np.array(sizes, np.uint32).tobytes()
        body = b''.join(data)
        return head + body

    def get_config(self) -> Dict[str, Any]:
        """Get object describing shard-writing configuration.

        Returns:
            Dict[str, Any]: JSON object.
        """
        obj = super().get_config()
        obj.update({
            'column_names': self.column_names,
            'column_encodings': self.column_encodings,
            'column_sizes': self.column_sizes
        })
        return obj

    def encode_joint_shard(self) -> bytes:
        """Encode a joint shard out of the cached samples (single file).

        Returns:
            bytes: File data.
        """
        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        offsets += len(num_samples.tobytes()) + len(offsets.tobytes()) + len(self.config_data)
        sample_data = b''.join(self.new_samples)
        return num_samples.tobytes() + offsets.tobytes() + self.config_data + sample_data
