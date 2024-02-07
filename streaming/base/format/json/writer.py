# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

""":class:`JSONWriter` writes samples to `.json` files that can be read by :class:`JSONReader`."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from streaming.base.format.base.writer import SplitWriter
from streaming.base.format.json.encodings import is_json_encoded, is_json_encoding

__all__ = ['JSONWriter']


class JSONWriter(SplitWriter):
    r"""Writes a streaming JSON dataset.

    Args:
        columns (Dict[str, str]): Sample columns.
        newline (str): Newline character inserted between samples. Defaults to ``\\n``.
        out (str | Tuple[str, str]): Output dataset directory to save shard files.

            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in the
               `local_dir` and also uploaded to a remote location.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (Union[int, str], optional): Optional shard size limit, after which point to
            start a new shard. If None, puts everything in one shard. Can specify bytes
            human-readable format as well, for example ``"100kb"`` for 100 kilobyte
            (100*1024) and so on. Defaults to ``1 << 26``.
        **kwargs (Any): Additional settings for the Writer.

            progress_bar (bool): Display TQDM progress bars for uploading output dataset files to
                a remote location. Default to ``False``.
            max_workers (int): Maximum number of threads used to upload output dataset files in
                parallel to a remote location. One thread is responsible for uploading one shard
                file to a remote location. Default to ``min(32, (os.cpu_count() or 1) + 4)``.
            exist_ok (bool): If the local directory exists and is not empty, whether to overwrite
                the content or raise an error. `False` raises an error. `True` deletes the
                content and starts fresh. Defaults to `False`.
    """

    format = 'json'

    def __init__(self,
                 *,
                 columns: Dict[str, str],
                 newline: str = '\n',
                 out: Union[str, Tuple[str, str]],
                 keep_local: bool = False,
                 compression: Optional[str] = None,
                 hashes: Optional[List[str]] = None,
                 size_limit: Optional[Union[int, str]] = 1 << 26,
                 **kwargs: Any) -> None:
        super().__init__(out=out,
                         keep_local=keep_local,
                         compression=compression,
                         hashes=hashes,
                         size_limit=size_limit,
                         **kwargs)
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
