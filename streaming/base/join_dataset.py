# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A StreamingDataset variant that supports joining on multiple Streams."""

import logging
import warnings
from math import ceil
from typing import Any, Dict, Iterator, Optional, Sequence, Union

from torch import distributed as dist
from torch.utils.data import IterableDataset

from streaming.base.array import Array
from streaming.base.dataset import (StreamingDataset)
from streaming.base.distributed import maybe_init_dist
from streaming.base.stream import Stream
from streaming.base.world import World
from streaming.base.util import bytes_to_int, number_abbrev_to_int

logger = logging.getLogger(__name__)

class StreamingJoinDataset(Array, IterableDataset):
    """A wrapper around multiple StreamingDatasets that joins samples on the fly.

    This class allows for joining multiple streams of data together, on a common
    sample id.

    Args:
        streams (Sequence[Stream], optional): One or more streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. This is required through checks. Defaults to ``None``.
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
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced
            across all streams. If ``None``, takes its value from the total number of underlying
            samples. Provide this field if you are weighting streams relatively to target a larger
            or smaller epoch size. Defaults to ``None``. Can also take in human-readable number
            abbreviations (e.g., ``"100k"``, ``"64M"``, ``"77b"``, etc). Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[int, str], optional): Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s)
            may be evicted (deleted from the local cache) in order to stay under the limit.
            Set to ``None`` to disable shard eviction. Supports integer bytes as well as string
            human-readable bytes (e.g., ``100b``, ``64kb``, ``77mb``, and so on). Defaults to
            ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``relaxed``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. The sample space is divided evenly according to the number of canonical
            nodes. The higher the value, the more independent non-overlapping paths the
            StreamingDataset replicas take through the shards per model replica (increasing data
            source diversity). If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.

            .. note::

                For sequential sample ordering, set ``shuffle`` to ``False`` and
                ``num_canonical_nodes`` to the number of physical nodes of the initial run.
        batch_size (int, optional): Per-device batch size, the same as what is passed to the
            DataLoader. This affects how the dataset is partitioned over the workers and is
            necessary for deterministic resumption and optimal performance. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``,
            ``per_stream``, or ``device_per_stream``. Defaults to ``random``.
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
        replication (int, optional): Determines how many consecutive devices will receive the same
            samples. Useful for training with tensor or sequence parallelism, where multiple
            devices need to see the same partition of the dataset. Defaults to ``None``.
    """

    def __init__(self,
                 *,
                 streams: Sequence[Stream] = None,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[Union[int, str]] = None,
                 predownload: Optional[int] = None,
                 cache_limit: Optional[Union[int, str]] = None,
                 sampling_method: str = 'balanced',
                 sampling_granularity: int = 1,
                 partition_algo: str = 'relaxed',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1e',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: Optional[int] = None,
                 batching_method: str = 'random',
                 allow_unsafe_types: bool = False,
                 replication: Optional[int] = None) -> None:
        
        # Global arguments (which do not live in Streams).
        self.predownload = predownload
        self.cache_limit = cache_limit
        self.sampling_method = sampling_method
        self.sampling_granularity = sampling_granularity
        self.partition_algo = partition_algo
        self.num_canonical_nodes = num_canonical_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_algo = shuffle_algo
        self.shuffle_seed = shuffle_seed
        self.shuffle_block_size = shuffle_block_size
        self.batching_method = batching_method
        self.allow_unsafe_types = allow_unsafe_types
        self.replication = replication

        # Initialize the World context.
        #   * This information is for the per-rank or per-worker process.
        #   * DataLoader worker processes may get a different worker ID and worker count than rank.
        #   * We save the rank Worlds here because we cannot instantiate a World inside our
        #     destructor.
        #   * `unique_` is who are for coordination purposes, where every process must be unique.
        #   * `parallel_` is who we think we are for iterating purposes, where groups of process
        #     must act the same if `replication` is specified.
        #     This can enable tensor or sequence parallelism.
        world = World.detect()
        self._unique_rank_world = world
        if replication is not None:
            self._parallel_rank_world = world.replicate(replication)
        else:
            self._parallel_rank_world = world.copy()
        self._unique_worker_world: World
        self._parallel_worker_world: World

        # Initialize initial_physical_nodes to None. If we are resuming, then we will set it to the
        # number of physical nodes of the initial run in the _resume function, or the number of
        # nodes specified in the `_parallel_rank_world` if using `replication`.
        self.initial_physical_nodes = None

        # Check streams vs remote/local.
        if bool(streams) == (bool(remote) or bool(local)):
            raise ValueError(
                'You must provide either `streams` or `remote`/`local`, but not both.')

        # Check sampling method is one of "balanced" or "fixed".
        if self.sampling_method not in ['balanced', 'fixed']:
            raise ValueError(
                f'Invalid sampling method: {sampling_method}. ' + \
                f'Must be one of `balanced` or `fixed`.'
            )

        # Check sampling granularity.
        if self.sampling_granularity <= 0:
            raise ValueError(f'`sampling_granularity` must be a positive integer, but got: ' +
                             f'{self.sampling_granularity}.')

        # Check batching method is one of "random", "stratified", "per_stream", or "device_per_stream".
        if self.batching_method not in ['random', 'stratified', 'per_stream', 'device_per_stream']:
            raise ValueError(
                f'Invalid batching method: {batching_method}. ' + \
                f'Must be one of `random`, `stratified`, `per_stream`, or `device_per_stream`.'
            )

        # issue deprecation warning for py1b shuffle algorithm.
        if self.shuffle_algo == 'py1b':
            warnings.warn('The \'py1b\' shuffle algorithm will soon be deprecated. \
                Please use the more performant \'py1br\' algorithm instead.',
                          DeprecationWarning,
                          stacklevel=2)

        # Check shuffle seed.
        if self.shuffle_seed < 0:
            raise ValueError(f'`shuffle_seed` must be a non-negative integer, but got: ' +
                             f'{self.shuffle_seed}.')

        # Check that predownload is at least per device batch size, and set it if currently `None`.
        if self.predownload is not None and self.batch_size is not None and \
            self.predownload < self.batch_size:
            warnings.warn(f'predownload < batch_size ({self.predownload} < {self.batch_size}).' +
                          f'This may result in slower batch time. Recommendation is to set ' +
                          f'predownload to at-least batch_size.')
        elif self.predownload is None:
            logger.warning(f'Because `predownload` was not specified, it will default to ' +
                           f'8*batch_size if batch_size is not None, otherwise 64. Prior to ' +
                           f'Streaming v0.7.0, `predownload` defaulted to ' +
                           f'max(batch_size, 256 * batch_size // num_canonical_nodes).')
            self.predownload = 8 * self.batch_size if self.batch_size is not None else 64

        # Convert epoch size from string to int, if needed. Cannot be negative.
        epoch_size_value = None
        if epoch_size:
            epoch_size_value = number_abbrev_to_int(epoch_size)
            if epoch_size_value < 0:
                raise ValueError(f'Epoch size cannot be negative. Received {epoch_size_value}.')

        # Initialize torch dist ourselves, if necessary.
        destroy_dist = maybe_init_dist()

        # Initialize the Stream defaults and normalize to a list of Streams.
        # >>> StreamingJoinDataset requires using the streams argument, since otherwise,
        # there is nothing to join.
        if streams and len(streams) >= 2:
            # >>> Each stream is handled by another StreamingDataset.
            self.sub_datasets = []
            default = {
                'remote': remote,
                'local': local,
                'split': split,
                'download_retry': download_retry,
                'download_timeout': download_timeout,
                'validate_hash': validate_hash,
                'keep_zip': keep_zip,
            }
            for stream in streams:
                stream.apply_default(default)
                sub_dataset = StreamingDataset(
                    streams=[stream],
                    epoch_size=epoch_size_value,
                    predownload=self.predownload,
                    cache_limit=self.cache_limit,
                    sampling_method=self.sampling_method,
                    sampling_granularity=self.sampling_granularity,
                    partition_algo=self.partition_algo,
                    num_canonical_nodes=self.num_canonical_nodes,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    shuffle_algo=self.shuffle_algo,
                    shuffle_seed=self.shuffle_seed,
                    shuffle_block_size=self.shuffle_block_size,
                    batching_method=self.batching_method,
                    allow_unsafe_types=self.allow_unsafe_types,
                    replication=self.replication,
                )
                self.sub_datasets.append(sub_dataset)
        elif len(streams) < 2:
            raise ValueError(f'Using StreamingJoinDataset requires passing in at least two data ' +
                             f'sources via `streams`. Received f{len(streams)} data streams. If ' +
                             f'you only have one data source, use StreamingDataset instead.')
        else:
            raise ValueError('Using StreamingJoinDataset requires passing in data sources via ' +
                             '`streams`.')

        # Set streams.
        self.streams = streams
        self.num_streams = len(streams)
        self.num_sub_datasets = len(self.sub_datasets)

        # Make sure that the number of samples matches across all sub_datasets.
        self.num_samples = self.sub_datasets[0].num_samples
        for sub_dataset_idx in range(1, len(self.sub_datasets)):
            sub_dataset = self.sub_datasets[sub_dataset_idx]
            if sub_dataset.num_samples != self.num_samples:
                raise ValueError(
                    f'Number of samples in each stream must match. ' +
                    f'Expected {self.num_samples}, but got {sub_dataset.num_samples} ' +
                    f'at sub_dataset index {sub_dataset_idx}.')

        # Check that cache limit is possible.
        if self.cache_limit:
            if isinstance(self.cache_limit, str):
                self.cache_limit = bytes_to_int(self.cache_limit)
            # Overall min cache usage should traverse all streams from all sub_datasets.
            min_cache_usage = 0
            for sub_dataset in self.sub_datasets:
                min_cache_usage += sum((stream.get_index_size() for stream in sub_dataset.streams))
            if self.cache_limit <= min_cache_usage:
                raise ValueError(f'Minimum cache usage ({min_cache_usage} bytes) is larger than ' +
                                 f'the cache limit ({self.cache_limit} bytes). Please raise ' +
                                 f'`cache_limit`. Recommendation is to provide a `cache_limit` ' +
                                 f'as high as possible to avoid thrashing.')
            # Warn user that they should set the cache limit to each sub_dataset's cache usage,
            # since tracking overall cache usage with shared memory from multiple sub_datasets gets
            # pretty ugly.
            warnings.warn('StreamingJoinDataset does not track overall cache usage, and each of ' +
                          'the sub-datasets will track their own cache usage. For the intended' +
                          'cache usage and eviction behavior, set the `cache_limit` to the ' +
                          'maximum intended cache usage of the largest sub-dataset.')

        # We make sure each sub_dataset has the same epoch size, and keep our epoch size
        # as that value.
        self.epoch_size = self.sub_datasets[0].epoch_size
        for sub_dataset_idx in range(1, len(self.sub_datasets)):
            sub_dataset = self.sub_datasets[sub_dataset_idx]
            if sub_dataset.epoch_size != self.epoch_size:
                raise ValueError(
                    f'Epoch size must match across all sub_datasets. ' +
                    f'Expected {self.epoch_size}, but got {sub_dataset.epoch_size} ' +
                    f'at sub_dataset index {sub_dataset_idx}.')

        # Length (__len__) is the resampled epoch size divided over the number of devices.
        self.length = ceil(self.epoch_size / self._parallel_rank_world.num_ranks)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if destroy_dist:
            dist.destroy_process_group()
    
    def state_dict(self, num_samples: int, from_beginning: bool) -> Dict[str, Any]:
        """Get the dataset state dict."""
        state_dict = self.sub_datasets[0].state_dict(num_samples, from_beginning)
        for sub_dataset_idx in range(1, len(self.sub_datasets)):
            ret_state_dict = self.sub_datasets[sub_dataset_idx].state_dict(num_samples, from_beginning)
            if ret_state_dict != state_dict:
                raise RuntimeError(
                    f'State dict must match across all sub_datasets. ' +
                    f'Expected {state_dict}, but got {ret_state_dict} ' +
                    f'at sub_dataset index {sub_dataset_idx}.')
        return state_dict
    
    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Loads the dataset state dict."""
        for sub_dataset in self.sub_datasets:
            sub_dataset.load_state_dict(obj)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate, returning joined samples from all streams."""

        # Initialize iterators for each sub_dataset.
        iterators = [iter(sub_dataset) for sub_dataset in self.sub_datasets]

        # Get the batch portion from each sub dataset
        while True:
            try:
                batch_parts = [next(iterator) for iterator in iterators]
                if '_id' not in batch_parts[0]:
                    raise ValueError(f'Missing join key "_id" in batch part. Make sure your ' +
                                     f'dataset parts are written with the join key "_id".')
                join_id = batch_parts[0]['_id']
                for batch_part in batch_parts:
                    if batch_part['_id'] != join_id:
                        raise ValueError(f'Sample IDs do not match across all sub_datasets. ' +
                                         f'Expected {join_id}, but got {batch_part["_id"]}.')
                    else:
                        # Join ID matches, so remove it from this batch part
                        batch_part.pop('_id')
                joined_batch = {k: v for batch_part in batch_parts for k, v in batch_part.items()}
                yield joined_batch
            except StopIteration:
                # Reached end of epoch.
                break


