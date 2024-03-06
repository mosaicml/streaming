# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A streaming WebVid dataset."""

import os
from time import sleep
from typing import Any, Optional

from streaming.base import StreamingDataset
from streaming.base.dataset import TICK, _Iterator
from streaming.base.storage import download_file


class StreamingInsideWebVid(StreamingDataset):
    """Streaming WebVid dataset.

    Videos are stored "inside" the shards, as is typically done.

    Args:
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            ``False``.
        epoch_size (int, optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value gets derived using per device batch size and number of
            canonical nodes ``max(batch_size, 256 * batch_size // num_canonical_nodes)``.
            Defaults to ``None``.
        cache_limit (int, optional): Maximum size in bytes of this StreamingDataset's shard cache.
            Before downloading a shard, the least recently used resident shard(s) may be evicted
            (deleted from the local cache) in order to stay under the limit. Set to ``None`` to
            disable shard eviction. Defaults to ``None``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. The sample space is divided evenly according to the number of canonical
            nodes. The higher the value, the more independent non-overlapping paths the
            StreamingDataset replicas take through the shards per model replica (increasing data
            source diversity). Defaults to ``None``, which is interpreted as 64 times the number
            of nodes of the initial run.

            .. note::

                For sequential sample ordering, set ``shuffle`` to ``False`` and
                ``num_canonical_nodes`` to the number of physical nodes of the initial run.
        batch_size (int, optional): Per-device batch size, the same as what is passed to the
            DataLoader. This affects how the dataset is partitioned over the workers and is
            necessary for deterministic resumption and optimal performance. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
    """

    def get_item(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().get_item(idx)
        # Processing goes here.
        return obj


class StreamingOutsideGIWebVid(StreamingDataset):
    """Streaming WebVid dataset.

    Videos are stored "outside" the shards, as a file per video. The extra download happens in
    get_item ("GI"), when samples are requested by the dataloader.

    Args:
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            ``False``.
        epoch_size (int, optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value gets derived using per device batch size and number of
            canonical nodes ``max(batch_size, 256 * batch_size // num_canonical_nodes)``.
            Defaults to ``None``.
        cache_limit (int, optional): Maximum size in bytes of this StreamingDataset's shard cache.
            Before downloading a shard, the least recently used resident shard(s) may be evicted
            (deleted from the local cache) in order to stay under the limit. Set to ``None`` to
            disable shard eviction. Defaults to ``None``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. The sample space is divided evenly according to the number of canonical
            nodes. The higher the value, the more independent non-overlapping paths the
            StreamingDataset replicas take through the shards per model replica (increasing data
            source diversity). Defaults to ``None``, which is interpreted as 64 times the number
            of nodes of the initial run.

            .. note::

                For sequential sample ordering, set ``shuffle`` to ``False`` and
                ``num_canonical_nodes`` to the number of physical nodes of the initial run.
        batch_size (int, optional): Per-device batch size, the same as what is passed to the
            DataLoader. This affects how the dataset is partitioned over the workers and is
            necessary for deterministic resumption and optimal performance. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
        extra_local (str, optional): Base destination of extra local sample downloads.
        extra_remote (str, optional): Base source of extra remote sample downloads.
    """

    def __init__(self,
                 *,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[int] = None,
                 predownload: Optional[int] = None,
                 cache_limit: Optional[int] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 extra_local: Optional[str] = None,
                 extra_remote: Optional[str] = None) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip,
                         epoch_size=epoch_size,
                         predownload=predownload,
                         cache_limit=cache_limit,
                         partition_algo=partition_algo,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed,
                         shuffle_block_size=shuffle_block_size)

        # Videos are stored outside of their shards here.
        self.download_timeout = download_timeout
        self.extra_local = extra_local
        self.extra_remote = extra_remote

    def get_item(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().get_item(idx)

        if self.extra_local and self.extra_remote:
            rel_path = obj['content_path']
            local = os.path.join(self.extra_local, rel_path)
            remote = os.path.join(self.extra_remote, rel_path)
            if not os.path.exists(local):
                download_file(remote, local, self.download_timeout)
            with open(local, 'rb') as fp:
                content = fp.read()
            obj['content'] = content

        # Processing goes here.

        return obj


class StreamingOutsideDTWebVid(StreamingDataset):
    """Streaming WebVid dataset.

    Videos are stored "outside" the shards, as a file per video. The extra download happens in
    _download_thread ("DT"), when the download thread prefetches the sample.

    Args:
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            ``False``.
        epoch_size (int, optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value gets derived using per device batch size and number of
            canonical nodes ``max(batch_size, 256 * batch_size // num_canonical_nodes)``.
            Defaults to ``None``.
        cache_limit (int, optional): Maximum size in bytes of this StreamingDataset's shard cache.
            Before downloading a shard, the least recently used resident shard(s) may be evicted
            (deleted from the local cache) in order to stay under the limit. Set to ``None`` to
            disable shard eviction. Defaults to ``None``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. The sample space is divided evenly according to the number of canonical
            nodes. The higher the value, the more independent non-overlapping paths the
            StreamingDataset replicas take through the shards per model replica (increasing data
            source diversity). Defaults to ``None``, which is interpreted as 64 times the number
            of nodes of the initial run.

            .. note::

                For sequential sample ordering, set ``shuffle`` to ``False`` and
                ``num_canonical_nodes`` to the number of physical nodes of the initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
        extra_local (str, optional): Base destination of extra local sample downloads.
        extra_remote (str, optional): Base source of extra remote sample downloads.
    """

    def __init__(self,
                 *,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[int] = None,
                 predownload: Optional[int] = None,
                 cache_limit: Optional[int] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 extra_local: Optional[str] = None,
                 extra_remote: Optional[str] = None) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip,
                         epoch_size=epoch_size,
                         predownload=predownload,
                         cache_limit=cache_limit,
                         partition_algo=partition_algo,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed,
                         shuffle_block_size=shuffle_block_size)

        # Videos are stored outside of their shards here.
        self.download_timeout = download_timeout
        self.extra_local = extra_local
        self.extra_remote = extra_remote

    def get_item(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().get_item(idx)

        if self.extra_local and self.extra_remote:
            rel_path = obj['content_path']
            local = os.path.join(self.extra_local, rel_path)
            remote = os.path.join(self.extra_remote, rel_path)
            if not os.path.exists(local):
                download_file(remote, local, self.download_timeout)
            with open(local, 'rb') as fp:
                content = fp.read()
            obj['content'] = content

        # Processing goes here.

        return obj

    def _download_thread(self, it: _Iterator) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling exit_threads() on its state (only one epoch is
        valid at a time).

        Each worker has its own download thread, which iterates ahead of the ready thread and yield
        loop.

        Args:
            it (_Iterator): State of __iter__.
        """
        # Download loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if it.should_exit():
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if it.prepare_index == it.total:
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = it.prepare_index - it.yield_index
                if self.predownload <= samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = it.sample_ids[it.prepare_index]
            if sample_id == -1:
                it.prepare_index += 1
                continue

            # Download and decompress the shard for this sample, if not already done.
            shard_id, _ = self.spanner[sample_id]
            self.prepare_shard(shard_id, False)

            # Predownload the sample's extra data.
            obj = super().get_item(sample_id)
            if self.extra_local and self.extra_remote:
                rel_path = obj['content_path']
                local = os.path.join(self.extra_local, rel_path)
                remote = os.path.join(self.extra_remote, rel_path)
                if not os.path.exists(local):
                    download_file(remote, local, self.download_timeout)

            # Step forward one sample.
            it.prepare_index += 1

        # Note that we exited.
        it.on_exit()
