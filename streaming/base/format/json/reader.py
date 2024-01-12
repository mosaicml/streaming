# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

""":class:`JSONReader` reads samples from `.json` files that were written by :class:`MDSWriter`."""

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from typing_extensions import Self

from streaming.base.format.base.reader import FileInfo, SplitReader

__all__ = ['JSONReader']


class JSONReader(SplitReader):
    """Provides random access to the samples of a JSON shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        newline (str): Newline character(s).
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard. Can specify bytes
            human-readable format as well, for example ``"100kb"`` for 100 kilobyte
            (100*1024) and so on.
        zip_data (FileInfo, optional): Compressed data file info.
        zip_meta (FileInfo, optional): Compressed meta file info.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        columns: Dict[str, str],
        compression: Optional[str],
        hashes: List[str],
        newline: str,
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        size_limit: Optional[Union[int, str]],
        zip_data: Optional[FileInfo],
        zip_meta: Optional[FileInfo],
    ) -> None:
        super().__init__(dirname, split, compression, hashes, raw_data, raw_meta, samples,
                         size_limit, zip_data, zip_meta)
        self.columns = columns
        self.newline = newline

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: Dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Args:
            dirname (str): Local directory containing shards.
            split (str, optional): Which dataset split to use, if any.
            obj (Dict[str, Any]): JSON object to load.

        Returns:
            Self: Loaded JSONReader.
        """
        args = deepcopy(obj)
        # Version check.
        if args['version'] != 2:
            raise ValueError(f'Unsupported streaming data version: {args["version"]}. ' +
                             f'Expected version 2.')
        del args['version']
        # Check format.
        if args['format'] != 'json':
            raise ValueError(f'Unsupported data format: {args["format"]}. ' +
                             f'Expected to be `json`.')
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'raw_meta', 'zip_data', 'zip_meta']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)

    def decode_sample(self, data: bytes) -> Dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        text = data.decode('utf-8')
        return json.loads(text)

    def get_sample_data(self, idx: int) -> bytes:
        """Get the raw sample data at the index.

        Args:
            idx (int): Sample index.

        Returns:
            bytes: Sample data.
        """
        meta_filename = os.path.join(self.dirname, self.split, self.raw_meta.basename)
        offset = (1 + idx) * 4
        with open(meta_filename, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
        data_filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
        with open(data_filename, 'rb', 0) as fp:
            fp.seek(begin)
            data = fp.read(end - begin)
        return data
