# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

""":class:`JSONWriter` writes samples to `.json` files that can be read by :class:`JSONReader`."""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from streaming.base.format.base.writer import SplitWriter
from streaming.base.format.json.encodings import is_json_encoded, is_json_encoding

__all__ = ['JSONWriter']


class JSONWriter(SplitWriter):
    r"""Writes a streaming JSON dataset.

    Args:
        columns (Dict[str, str]): Sample columns.
        newline (str): Newline character inserted between samples. Defaults to ``\\n``.
        local: (str, optional): Optional local output dataset directory. If not provided, a random
            temp directory will be used. If ``remote`` is provided, this is where shards are cached
            before uploading. One or both of ``local`` and ``remote`` must be provided. Defaults to
            ``None``.
        remote: (str, optional): Optional remote output dataset directory. If not provided, no
            uploading will be done. Defaults to ``None``.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Defaults to ``None``.
    """

    format = 'json'

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 newline: str = '\n',
                 local: Optional[str] = None,
                 remote: Optional[str] = None,
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[int] = 1 << 26) -> None:
        super().__init__(local=local,
                         remote=remote,
                         keep_local=keep_local,
                         compression=compression,
                         hashes=hashes,
                         size_limit=size_limit)
        for encoding in columns.values():
            assert is_json_encoding(encoding)

        self.columns = columns
        self.newline = newline

    def encode_sample(self, sample: Dict[str, Any]) -> bytes:
        """Encode a sample dict to bytes.

        Args:
            sample (Dict[str, Any]): Sample dict.

        Returns:
            bytes: Sample encoded as bytes.
        """
        obj = {}
        for key, encoding in self.columns.items():
            value = sample[key]
            assert is_json_encoded(encoding, value)
            obj[key] = value
        text = json.dumps(obj, sort_keys=True) + self.newline
        return text.encode('utf-8')

    def get_config(self) -> Dict[str, Any]:
        """Get object describing shard-writing configuration.

        Returns:
            Dict[str, Any]: JSON object.
        """
        obj = super().get_config()
        obj.update({'columns': self.columns, 'newline': self.newline})
        return obj

    def encode_split_shard(self) -> Tuple[bytes, bytes]:
        """Encode a split shard out of the cached samples (data, meta files).

        Returns:
            Tuple[bytes, bytes]: Data file, meta file.
        """
        data = b''.join(self.new_samples)

        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        obj = self.get_config()
        text = json.dumps(obj, sort_keys=True)
        meta = num_samples.tobytes() + offsets.tobytes() + text.encode('utf-8')

        return data, meta
