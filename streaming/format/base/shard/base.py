# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming shard abstract base classes."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set

from typing_extensions import Self

from streaming.array import Array
from streaming.format.base.file import ShardFile
from streaming.stream.dir_conf import StreamDirConf

__all__ = ['Shard']


class Shard(Array):
    """Streaming shard abstract base class.

    Args:
        conf (Any, optional): JSON shard config. Defaults to ``None``.
        stream (StreamDirConf): Link back up to the Stream that owns this shard, from which we
            get arguments which are shared across all shards like remote/local paths. Avoids an
            import cycle by Stream subclassing StreamDirConf.
        num_samples (int): Number of samples in this shard.
    """

    def __init__(
        self,
        *,
        conf: Optional[Any] = None,
        stream: StreamDirConf,
        num_samples: int,
        files: List[ShardFile],
    ) -> None:
        self.conf = conf
        self.stream = stream
        self.num_samples = self.samples = num_samples
        self.files = files

    @classmethod
    @abstractmethod
    def from_json(cls, stream: StreamDirConf, obj: Dict[str, Any]) -> Self:
        """Initialize a Shard from this configuration.

        Args:
            stream (StreamDirConf): Owning Stream.
            obj (Dict[str, Any]): JSON object.

        Returns:
            Self: Instance of this class.
        """
        raise NotImplementedError

    def validate(self) -> None:
        """Check whether this shard is acceptable to be part of some Stream."""
        for file in self.files:
            file.validate()

    def __len__(self) -> int:
        """Get the number of samples in this shard.

        Returns:
            int: Sample count.
        """
        return self.num_samples

    def set_stream(self, stream: StreamDirConf) -> None:
        """Save a reference to the owning Stream, as many Stream args apply to all its Shards.

        Args:
            stream (StreamDirConf): The Stream that owns this Shard.
        """
        self.stream = stream
        for file in self.files:
            file.set_stream(stream)

    def inventory_local(self, listing: Set[str]) -> Optional[int]:
        """Normalize what files/phases of files are present to a coherent state.

        This is used to initialize with a local cache directory that already contains shard files.
        Make sure they are the correct files, and that which ones exist are coherent for our
        purposes, or that they are deleted so that we are in a coherent filesystem state.

        Args:
            listing (Set[str]): Relative paths to all the files under the dataset root, which is
                done just once in order to avoid hammering the filesystem.

        Returns:
            Optional[int]: Cache usage if present, or None if not present, after normalization.
        """
        cache_usage = 0
        for file in self.files:
            file_cache_usage = file.inventory_local(listing)
            if file_cache_usage is None:
                self.evict()
                cache_usage = 0
                break
            cache_usage += file_cache_usage
        return cache_usage

    def fetch(self) -> int:
        """Download and/or unzip and/or canonicalize this shard to being ready for use.

        Returns:
            int: Change in cache usage, in bytes.
        """
        cache_usage_change = 0
        for file in self.files:
            cache_usage_change += file.fetch()
        return cache_usage_change

    def evict(self) -> int:
        """Delete all of this shard's files in any of their phases.

        Returns:
            int: Change in cache usage, in bytes.
        """
        cache_usage_change = 0
        for file in self.files:
            cache_usage_change += file.evict()
        return cache_usage_change

    @property
    def size(self) -> int:
        """Get the number of samples in this shard.

        Note: the ``Array`` abstract class, from which we inherit, gets its concept of length from
        ``self.size`` instead of ``self.__len__``. This is because ``IterableDataset.__len__``,
        from which ``StreamingDataset`` inherits, means ``ceil(epoch_size / ranks)``. That in turn
        is because with ``IterableDataset``s you just have a potentially infinite stream of
        individual samples and no way to a priori tell when you will be done otherwise.

        Returns:
            int: Sample count.
        """
        return self.num_samples

    @abstractmethod
    def get_item(self, index: int) -> Dict[str, Any]:
        """Get the sample at the index.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, Any]: Sample dict.
        """
        raise NotImplementedError
