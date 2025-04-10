# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

""":class:`MDSReader` reads samples in `.mds` files written by :class:`StreamingDatasetWriter`."""

import os
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
from typing_extensions import Self

from streaming.base.format.base.reader import FileInfo, JointReader
from streaming.base.format.mds.encodings import is_mds_encoding_safe, mds_decode

__all__ = ['MDSReader']


class MDSReader(JointReader):
    """Provides random access to the samples of an MDS shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        column_encodings (List[str]): Column encodings.
        column_names (List[str]): Column names.
        column_sizes (List[Optional[int]]): Column fixed sizes, if any.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        raw_data (FileInfo): Uncompressed data file info.
        samples (int): Number of samples in this shard.
        size_limit (Union[int, str], optional): Optional shard size limit, after
            which point to start a new shard. If None, puts everything in one shard.
            Can specify bytes in human-readable format as well, for example
            ``"100kb"`` for 100 kilobyte (100*1024) and so on.
        zip_data (FileInfo, optional): Compressed data file info.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        column_encodings: list[str],
        column_names: list[str],
        column_sizes: list[Optional[int]],
        compression: Optional[str],
        hashes: list[str],
        raw_data: FileInfo,
        samples: int,
        size_limit: Optional[Union[int, str]],
        zip_data: Optional[FileInfo],
    ) -> None:
        super().__init__(dirname, split, compression, hashes, raw_data, samples, size_limit,
                         zip_data)
        self.column_encodings = column_encodings
        self.column_names = column_names
        self.column_sizes = column_sizes

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Args:
            dirname (str): Local directory containing shards.
            split (str, optional): Which dataset split to use, if any.
            obj (Dict[str, Any]): JSON object to load.

        Returns:
            Self: Loaded MDSReader.
        """
        args = deepcopy(obj)
        args_version = args['version']
        if args_version != 2:
            raise ValueError(
                f'Unsupported streaming data version: {args_version}. Expected version 2.')
        del args['version']
        args_format = args['format']
        if args_format != 'mds':
            raise ValueError(f'Unsupported data format: {args_format}. Expected to be `mds`.')
        del args['format']
        args['dirname'] = dirname
        args['split'] = split
        for key in ['raw_data', 'zip_data']:
            arg = args[key]
            args[key] = FileInfo(**arg) if arg else None
        return cls(**args)

    def validate(self, allow_unsafe_types: bool) -> None:
        """Check whether this shard is acceptable to be part of some Stream.

        Args:
            allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
                execution during deserialization, whether to keep going if ``True`` or raise an
                error if ``False``.
        """
        if not allow_unsafe_types:
            for column_id, encoding in enumerate(self.column_encodings):
                if not is_mds_encoding_safe(encoding):
                    name = self.column_names[column_id]
                    raise ValueError(f'Column {name} contains an unsafe type: {encoding}. To ' +
                                     f'proceed anyway, set ``allow_unsafe_types=True``.')

    def decode_sample(self, data: bytes) -> dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        sizes = []
        idx = 0
        for key, size in zip(self.column_names, self.column_sizes):
            if size:
                sizes.append(size)
            else:
                size, = np.frombuffer(data[idx:idx + 4], np.uint32)
                sizes.append(size)
                idx += 4
        sample = {}
        for key, encoding, size in zip(self.column_names, self.column_encodings, sizes):
            value = data[idx:idx + size]
            sample[key] = mds_decode(encoding, value)
            idx += size
        return sample

    def get_sample_data(self, idx: int) -> bytes:
        """Get the raw sample data at the index.

        Args:
            idx (int): Sample index.

        Returns:
            bytes: Sample data.
        """
        filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
        offset = (1 + idx) * 4
        with open(filename, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
            fp.seek(begin)
            data = fp.read(end - begin)
        if not data:
            raise IndexError(
                f'Relative sample index {idx} is not present in the {self.raw_data.basename} file.'
            )
        return data
