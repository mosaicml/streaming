# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Read and decode sample from shards."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set

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

    def _evict_raw(self) -> None:
        """Remove all raw files belonging to this shard."""
        for raw_info, _ in self.file_pairs:
            filename = os.path.join(self.dirname, self.split, raw_info.basename)
            if os.path.exists(filename):
                os.remove(filename)

    def _evict_zip(self) -> None:
        """Remove all zip files belonging to this shard."""
        for _, zip_info in self.file_pairs:
            if zip_info:
                filename = os.path.join(self.dirname, self.split, zip_info.basename)
                if os.path.exists(filename):
                    os.remove(filename)

    def evict(self) -> None:
        """Remove all files belonging to this shard."""
        self._evict_raw()
        self._evict_zip()

    def init_local_dir(self, filenames_present: Set[str], keep_zip: bool) -> bool:
        """Bring what shard files are present to a consistent state, returning whether present.

        Args:
            filenames_present (Set[str]): The listing of all files under dirname/[split/]. This is
                listed once and then saved because there could potentially be very many shard
                files.
            keep_zip (bool): Whether to keep zip files when decompressing. Possible when
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
                if filename in filenames_present:
                    raw_files_present += 1
            if zip_info:
                filename = os.path.join(self.dirname, self.split, zip_info.basename)
                if filename in filenames_present:
                    zip_files_present += 1

        # If the shard raw files are partially present, garbage collect the present ones and mark
        # the shard raw as not present, in order to achieve consistency.
        if not raw_files_present:
            is_raw_present = False
        elif raw_files_present < len(self.file_pairs):
            is_raw_present = False
            self._evict_raw()
        else:
            is_raw_present = True

        # Same as the above, but for shard zip files.
        if not zip_files_present:
            is_zip_present = False
        elif zip_files_present < len(self.file_pairs):
            is_zip_present = False
            self._evict_zip()
        else:
            is_zip_present = True

        # Do we keep_zip?
        if keep_zip:
            # If we can keep_zip, and we do, and have either raw or zip, we must have the other one
            # too, because they are downloaded and decompressed together.
            if self.compression and (is_zip_present != is_raw_present):
                if is_raw_present:
                    is_raw_present = False
                    self._evict_raw()
                elif is_zip_present:
                    is_zip_present = False
                    self._evict_zip()
        else:
            # If we don't keep_zip, drop any zip files.
            if is_zip_present:
                is_zip_present = False
                self._evict_zip()

        # Now, the shard is either entirely or not at all present given keep_zip.
        return is_raw_present

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

    def get_full_size(self) -> int:
        """Get the full size of this shard.

        "Full" in this case means both the raw (decompressed) and zip (compressed) versions are
        resident (assuming it has a zip form). This is the maximum disk usage the shard can reach.
        When compressed was used, even if keep_zip is ``False``, the zip form must still be
        resident at the same time as the raw form during shard decompression.

        Returns:
            int: Size in bytes.
        """
        raw_size = self.get_raw_size()
        zip_size = self.get_zip_size() or 0
        return raw_size + zip_size

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
                size = self.get_full_size()
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
