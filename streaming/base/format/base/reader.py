# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Read and decode sample from shards."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Union

from streaming.base.array import Array
from streaming.base.util import bytes_to_int

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


class Reader(Array, ABC):
    """Provides random access to the samples of a shard.

    Args:
        dirname (str): Local dataset directory.
        split (str, optional): Which dataset split to use, if any.
        compression (str, optional): Optional compression or compression:level.
        hashes (List[str]): Optional list of hash algorithms to apply to shard files.
        samples (int): Number of samples in this shard.
        size_limit (Union[int, str], optional): Optional shard size limit, after which
            point to start a new shard. If None, puts everything in one shard. Can
            specify bytes in human-readable format as well, for example ``"100kb"``
            for 100 kilobyte (100*1024) and so on.
    """

    def __init__(
        self,
        dirname: str,
        split: Optional[str],
        compression: Optional[str],
        hashes: List[str],
        samples: int,
        size_limit: Optional[Union[int, str]],
    ) -> None:

        if size_limit:
            if (isinstance(size_limit, str)):
                size_limit = bytes_to_int(size_limit)
            if size_limit < 0:
                raise ValueError(f'`size_limit` must be greater than zero, instead, ' +
                                 f'found as {size_limit}.')

        self.dirname = dirname
        self.split = split or ''
        self.compression = compression
        self.hashes = hashes
        self.samples = samples
        self.size_limit = size_limit

        self.file_pairs = []

    def validate(self, allow_unsafe_types: bool) -> None:
        """Check whether this shard is acceptable to be part of some Stream.

        Args:
            allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
                execution during deserialization, whether to keep going if ``True`` or raise an
                error if ``False``.
        """
        pass

    @property
    def size(self):
        """Get the number of samples in this shard.

        Returns:
            int: Sample count.
        """
        return self.samples

    def __len__(self) -> int:
        """Get the number of samples in this shard.

        Returns:
            int: Sample count.
        """
        return self.samples

    def _evict_raw(self) -> int:
        """Remove all raw files belonging to this shard.

        Returns:
            int: Bytes evicted from cache.
        """
        size = 0
        for raw_info, _ in self.file_pairs:
            filename = os.path.join(self.dirname, self.split, raw_info.basename)
            if os.path.exists(filename):
                os.remove(filename)
                size += raw_info.bytes
        return size

    def _evict_zip(self) -> int:
        """Remove all zip files belonging to this shard.

        Returns:
            int: Bytes evicted from cache.
        """
        size = 0
        for _, zip_info in self.file_pairs:
            if zip_info:
                filename = os.path.join(self.dirname, self.split, zip_info.basename)
                if os.path.exists(filename):
                    os.remove(filename)
                    size += zip_info.bytes
        return size

    def evict(self) -> int:
        """Remove all files belonging to this shard.

        Returns:
            int: Bytes evicted from cache.
        """
        return self._evict_raw() + self._evict_zip()

    def set_up_local(self, listing: Set[str], safe_keep_zip: bool) -> int:
        """Bring what shard files are present to a consistent state, returning whether present.

        Args:
            listing (Set[str]): The listing of all files under dirname/[split/]. This is listed
                once and then saved because there could potentially be very many shard files.
            safe_keep_zip (bool): Whether to keep zip files when decompressing. Possible when
                compression was used. Necessary when local is the remote or there is no remote.

        Returns:
            bool: Whether the shard is present.
        """
        # For raw/zip to be considered present, each raw/zip file must be present.
        raw_files_present = 0
        zip_files_present = 0
        for raw_info, zip_info in self.file_pairs:
            if raw_info:
                filename = os.path.join(self.dirname, self.split, raw_info.basename)
                if filename in listing:
                    raw_files_present += 1
            if zip_info:
                filename = os.path.join(self.dirname, self.split, zip_info.basename)
                if filename in listing:
                    zip_files_present += 1

        # If the shard raw files are partially present, garbage collect the present ones and mark
        # the shard raw as not present, in order to achieve consistency.
        if not raw_files_present:
            has_raw = False
        elif raw_files_present < len(self.file_pairs):
            has_raw = False
            self._evict_raw()
        else:
            has_raw = True

        # Same as the above, but for shard zip files.
        if not zip_files_present:
            has_zip = False
        elif zip_files_present < len(self.file_pairs):
            has_zip = False
            self._evict_zip()
        else:
            has_zip = True

        # Enumerate cases of raw/zip presence.
        if self.compression:
            if safe_keep_zip:
                if has_raw:
                    if has_zip:
                        # Present (normalized).
                        pass
                    else:
                        # Missing: there is no natural way to arrive at this state, so drop raw.
                        has_raw = False
                        self._evict_raw()
                else:
                    if has_zip:
                        # Present: but missing raw, so need to decompress upon use.
                        pass
                    else:
                        # Missing (normalized).
                        pass
            else:
                if has_raw:
                    if has_zip:
                        # Present: zip is unnecessary, so evict it.
                        has_zip = False
                        self._evict_raw()
                    else:
                        # Present (normalized).
                        pass
                else:
                    if has_zip:
                        # Present: but missing raw, so need to decompress and evict zip upon use.
                        pass
                    else:
                        # Missing (normalized).
                        pass
        else:
            if has_zip:
                raise ValueError('Shard is invalid: compression was not used, but has a ' +
                                 'compressed form.')

        # Get cache usage. Shard is present if either raw or zip are present.
        size = 0
        if has_raw:
            size += self.get_raw_size()
        if has_zip:
            size += self.get_zip_size() or 0
        return size

    def get_raw_size(self) -> int:
        """Get the raw (uncompressed) size of this shard.

        Returns:
            int: Size in bytes.
        """
        size = 0
        for info, _ in self.file_pairs:
            size += info.bytes
        return size

    def get_zip_size(self) -> Optional[int]:
        """Get the zip (compressed) size of this shard, if compression was used.

        Returns:
            Optional[int]: Size in bytes, or ``None`` if does not exist.
        """
        size = 0
        for _, info in self.file_pairs:
            if info is None:
                return None
            size += info.bytes
        return size

    def get_max_size(self) -> int:
        """Get the full size of this shard.

        "Max" in this case means both the raw (decompressed) and zip (compressed) versions are
        resident (assuming it has a zip form). This is the maximum disk usage the shard can reach.
        When compressed was used, even if keep_zip is ``False``, the zip form must still be
        resident at the same time as the raw form during shard decompression.

        Returns:
            int: Size in bytes.
        """
        return self.get_raw_size() + (self.get_zip_size() or 0)

    def get_persistent_size(self, keep_zip: bool) -> int:
        """Get the persistent size of this shard.

        "Persistent" in this case means whether both raw and zip are present is subject to
        keep_zip. If we are not keeping zip files after decompression, they don't count to the
        shard's persistent size on disk.

        Args:
            keep_zip (bool): Whether to keep zip files after decompressing.

        Returns:
            int: Size in bytes.
        """
        if self.compression:
            if keep_zip:
                size = self.get_max_size()
            else:
                size = self.get_raw_size()
        else:
            size = self.get_raw_size()
        return size

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

    def get_item(self, idx: int) -> Dict[str, Any]:
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
        size_limit (Union[int, str], optional): Optional shard size limit, after which
        point to start a new shard. If None, puts everything in one shard.
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
        size_limit: Optional[Union[int, str]],
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
        size_limit (Union[int, str], optional): Optional shard size limit, after which
            point to start a new shard. If None, puts everything in one shard.
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
        size_limit: Optional[Union[int, str]],
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
