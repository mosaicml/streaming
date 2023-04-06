# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A streaming WebVid dataset."""

import os
from time import sleep
from typing import Any, Optional

from streaming.base import StreamingDataset
from streaming.base.dataset import TICK, _IterState
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
            `False``.
        keep_raw (bool): Whether to keep or delete the decompressed form (or only form)
            of shards after all their samples have been yielded this epoch. If ``False``, keep iff
            remote is local or no remote and no compression. Defaults to ``True``.
        samples_per_epoch (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
    """

    def __getitem__(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().__getitem__(idx)
        # Processing goes here.
        return obj


class StreamingOutsideGIWebVid(StreamingDataset):
    """Streaming WebVid dataset.

    Videos are stored "outside" the shards, as a file per video. The extra download happens in
    __getitem__ ("GI"), when samples are requested by the dataloader.

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
            `False``.
        keep_raw (bool): Whether to keep or delete the decompressed form (or only form)
            of shards after all their samples have been yielded this epoch. If ``False``, keep iff
            remote is local or no remote and no compression. Defaults to ``True``.
        samples_per_epoch (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
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
                 keep_raw: bool = True,
                 samples_per_epoch: Optional[int] = None,
                 predownload: Optional[int] = 100_000,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 extra_local: Optional[str] = None,
                 extra_remote: Optional[str] = None) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip,
                         keep_raw=keep_raw,
                         samples_per_epoch=samples_per_epoch,
                         predownload=predownload,
                         partition_algo=partition_algo,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed)

        # Videos are stored outside of their shards here.
        self.download_timeout = download_timeout
        self.extra_local = extra_local
        self.extra_remote = extra_remote

    def __getitem__(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().__getitem__(idx)

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
            `False``.
        keep_raw (bool): Whether to keep or delete the decompressed form (or only form)
            of shards after all their samples have been yielded this epoch. If ``False``, keep iff
            remote is local or no remote and no compression. Defaults to ``True``.
        samples_per_epoch (int, optional): Provide this field iff you are weighting sub-datasets
            proportionally. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1s``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
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
                 keep_raw: bool = True,
                 samples_per_epoch: Optional[int] = None,
                 predownload: Optional[int] = 100_000,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 extra_local: Optional[str] = None,
                 extra_remote: Optional[str] = None) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip,
                         keep_raw=keep_raw,
                         samples_per_epoch=samples_per_epoch,
                         predownload=predownload,
                         partition_algo=partition_algo,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed)

        # Videos are stored outside of their shards here.
        self.download_timeout = download_timeout
        self.extra_local = extra_local
        self.extra_remote = extra_remote

    def __getitem__(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().__getitem__(idx)

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

    def _download_thread(self, state: _IterState) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling stop() on its state (only one epoch is valid at a
        time).

        Each worker has its own download thread, which iterates ahead of the main thread.

        Args:
            state (_IterState): The partition state.
        """
        # Download loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if state.is_stopped:
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if state.download_index == state.total:
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = state.download_index - state.yield_index
                if self.predownload <= samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = state.sample_ids[state.download_index]
            if sample_id == -1:
                state.download_index += 1
                continue

            # Download and decompress the shard for this sample, if not already done.
            shard_id, _ = self.index.find_sample(sample_id)
            self._download_or_skip_shard(shard_id, False)

            # Predownload the sample's extra data.
            obj = super().__getitem__(sample_id)
            if self.extra_local and self.extra_remote:
                rel_path = obj['content_path']
                local = os.path.join(self.extra_local, rel_path)
                remote = os.path.join(self.extra_remote, rel_path)
                if not os.path.exists(local):
                    download_file(remote, local, self.download_timeout)

            state.download_index += 1
