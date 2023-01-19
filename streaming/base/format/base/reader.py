# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Read and decode sample from shards."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

__all__ = ['FileInfo', 'Reader', 'JointReader', 'SplitReader']


@dataclass
class FileInfo(object):
    """File validation info.

    Args:
        basename (str): File basename.
        bytes (int): File size in bytes.
        hashes (Dict[str, str]): Mapping of hash algorithm to hash value.
    """
    basename: str
    bytes: int
    hashes: Dict[str, str]


class Reader(ABC):
    """Provides random access to the samples of a shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        samples (int): Number of samples in this shard.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        compression: Optional[str],
        hashes: List[str],
        samples: int,
        size_limit: Optional[int],
    ) -> None:
        self.dirname = dirname
        self.split = split or ''
        self.compression = compression
        self.hashes = hashes
        self.samples = samples
        self.size_limit = size_limit

        self.file_pairs = []

    def __len__(self) -> int:
        """Get the number of samples in this shard.

        Returns:
            int: Sample count.
        """
        return self.samples

    @abstractmethod
    def decode_sample(self, data: bytes) -> Dict[str, Any]:
        """Decode a sample dict from bytes.

        Args:
            data (bytes): The sample encoded as bytes.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_data(self, idx: int) -> bytes:
        """Get the raw sample data at the index.

        Args:
            idx (int): Sample index.

        Returns:
            bytes: Sample data.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        data = self.get_sample_data(idx)
        return self.decode_sample(data)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the samples of this shard.

        Returns:
            Iterator[Dict[str, Any]]: Iterator over samples.
        """
        for i in range(len(self)):
            yield self[i]


class JointReader(Reader):
    """Provides random access to the samples of a joint shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        raw_data (FileInfo): Uncompressed data file info.
        samples (int): Number of samples in this shard.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
        zip_data (FileInfo, optional): Compressed data file info.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        compression: Optional[str],
        hashes: List[str],
        raw_data: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: Optional[FileInfo],
    ) -> None:
        super().__init__(dirname, split, compression, hashes, samples, size_limit)
        self.raw_data = raw_data
        self.zip_data = zip_data
        self.file_pairs.append((raw_data, zip_data))


class SplitReader(Reader):
    """Provides random access to the samples of a split shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        raw_data (FileInfo): Uncompressed data file info.
        raw_meta (FileInfo): Uncompressed meta file info.
        samples (int): Number of samples in this shard.
        size_limit (int, optional): Optional shard size limit, after which point to start a new
            shard. If None, puts everything in one shard.
        zip_data (FileInfo, optional): Compressed data file info.
        zip_meta (FileInfo, optional): Compressed meta file info.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        compression: Optional[str],
        hashes: List[str],
        raw_data: FileInfo,
        raw_meta: FileInfo,
        samples: int,
        size_limit: Optional[int],
        zip_data: Optional[FileInfo],
        zip_meta: Optional[FileInfo],
    ) -> None:
        super().__init__(dirname, split, compression, hashes, samples, size_limit)
        self.raw_data = raw_data
        self.raw_meta = raw_meta
        self.zip_data = zip_data
        self.zip_meta = zip_meta
        self.file_pairs.append((raw_meta, zip_meta))
        self.file_pairs.append((raw_data, zip_data))
