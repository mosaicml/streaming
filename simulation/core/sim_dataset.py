# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Near replica of StreamingDataset for simulation purposes."""

import logging
import os
import shutil
import time
import warnings
from math import ceil
from typing import Optional, Sequence, Union

import numpy as np
from core.sim_spanner import SimulationSpanner
from core.sim_world import SimulationWorld
from numpy.typing import NDArray

from streaming.base import Stream, StreamingDataset
from streaming.base.batching import generate_work
from streaming.base.format import get_index_basename
from streaming.base.spanner import Spanner
from streaming.base.util import bytes_to_int, number_abbrev_to_int

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SimulationDataset(StreamingDataset):
    """Near replica of StreamingDataset for simulation purposes.

    Args:
        nodes (int): Number of nodes.
        devices (int): Number of devices.
        workers (int): Number of workers.
        streams (Optional[Sequence[Stream]]): One or more streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (Optional[str]): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (Optional[str]): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (Optional[str]): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (Optional[str]): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            ``False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``. Can also take in human-readable number
            abbreviations (e.g., ``"100k"``, ``"64M"``, ``"77b"``, and so on). Defaults to ``None``.
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
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
    """

    def __init__(self,
                 nodes: int,
                 devices: int,
                 workers: int,
                 streams: Optional[Sequence[Stream]] = None,
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
                 partition_algo: str = 'relaxed',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1e',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: Optional[int] = None,
                 sampling_method: str = 'balanced',
                 sampling_granularity: int = 1,
                 batching_method: str = 'random',
                 allow_unsafe_types: bool = False) -> None:

        # Time how long it takes for StreamingDataset instantiation
        t0 = time.time()

        # Global arguments (which do not live in Streams).
        self.nodes = nodes
        self.devices = devices
        self.workers = workers
        self.cache_limit = cache_limit
        self.partition_algo = partition_algo
        self.predownload = predownload
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_algo = shuffle_algo
        self.shuffle_seed = shuffle_seed
        self.shuffle_block_size = shuffle_block_size
        self.sampling_method = sampling_method
        self.sampling_granularity = sampling_granularity
        self.batching_method = batching_method
        self.num_canonical_nodes = num_canonical_nodes
        self.allow_unsafe_types = allow_unsafe_types

        self.initial_physical_nodes = nodes

        # Set num_canonical_nodes based on the shuffling algorithm chosen.
        if self.num_canonical_nodes is None:
            if self.shuffle_algo in ['py1s', 'py2s']:
                self.num_canonical_nodes = 64 * self.nodes
            else:
                self.num_canonical_nodes = self.nodes

        # Set shuffle_block_size if not provided, based on num_canonical_nodes.
        if self.shuffle_block_size is None:
            self.shuffle_block_size = max(4_000_000 // self.num_canonical_nodes, 1 << 18)

        # Check streams vs remote/local.
        if bool(streams) == (bool(remote) or bool(local)):
            raise ValueError(
                'You must provide either `streams` or `remote`/`local`, but not both.')

        # Check sampling method is one of "balanced" or "fixed".
        if self.sampling_method not in ['balanced', 'fixed']:
            raise ValueError(
                f'Invalid sampling method: {sampling_method}. Must be one of `balanced` or `fixed`.'
            )

        # Check sampling method is one of "balanced" or "fixed".
        if self.batching_method not in ['random', 'per_stream', 'stratified']:
            raise ValueError(
                f'Invalid batching method: {batching_method}. Must be one of `random`, \
                    `per_stream`, or `stratified`.')

        # Check that predownload is at least per device batch size, and set it if currently `None`.
        if self.predownload is not None and self.batch_size is not None and \
            self.predownload < self.batch_size:
            warnings.warn(f'predownload < batch_size ({self.predownload} < {self.batch_size}).' +
                          f'This may result in slower batch time. Recommendation is to set ' +
                          f'predownload to at-least batch_size.')
        elif self.predownload is None:
            self.predownload = 8 * self.batch_size if self.batch_size is not None else 64

        self.batch_size = batch_size

        # Convert epoch size from string to int, if needed. Cannot be negative.
        epoch_size_value = None
        if epoch_size:
            epoch_size_value = number_abbrev_to_int(epoch_size)
            if epoch_size_value < 0:
                raise ValueError(f'Epoch size cannot be negative. Received {epoch_size_value}.')

        # Initialize the Stream defaults and normalize to a list of Streams.
        if streams:
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
        else:
            default = Stream(remote=remote,
                             local=local,
                             split=split,
                             download_retry=download_retry,
                             download_timeout=download_timeout,
                             validate_hash=validate_hash,
                             keep_zip=keep_zip)
            streams = [default]

        # Validate the stream weighting scheme (relative or absolute) to catch errors before we go
        # to the trouble of loading them.
        Stream.validate_weights(streams)

        # Set streams.
        self.streams = streams
        self.num_streams = len(streams)

        self.stream_info = {}
        # 0 means index file is remote, 1 means local, 2 means created
        indices_created = []
        for stream_idx, stream in enumerate(self.streams):
            if stream.remote:
                filepath = os.path.join(stream.remote, stream.split, get_index_basename())
                indices_created.append(0)
            else:
                filepath = os.path.join(stream.local, stream.split, get_index_basename())
                # This suffix means a mock index file was created. Have to clean up later.
                if stream.local.split('_')[-1] == 'indexcreated':
                    indices_created.append(2)
                else:
                    # Index file is local. Don't delete later.
                    indices_created.append(1)
            self.stream_info[stream_idx] = {
                'path': filepath,
                'local': stream.local,
                'remote': stream.remote,
                'proportion': stream._proportion,
                'repeat': stream._repeat,
                'choose': stream._choose
            }

        # Initialize the SimulationWorld, which tells us about nodes/devices/workers
        self.world = SimulationWorld(self.nodes, self.devices, self.workers)

        # Download each stream's index, load their shards, and map streams <-> shards.
        self.num_samples = 0
        self.shards = []
        stream_per_shard = []
        self.shard_offset_per_stream = np.zeros(self.num_streams, np.int64)
        self.shards_per_stream = np.zeros(self.num_streams, np.int64)
        self.sample_offset_per_stream = np.zeros(self.num_streams, np.int64)
        self.samples_per_stream = np.zeros(self.num_streams, np.int64)
        index_filenames = []
        local_foldernames = []
        for stream_id, stream in enumerate(self.streams):
            logger.info(f' Processing index file for stream {stream_id + 1}')
            stream_shards = stream.get_shards(self.world, self.allow_unsafe_types)
            num_stream_samples = sum(map(len, stream_shards))
            index_filename = os.path.join(stream.local, stream.split, get_index_basename())
            index_filenames.append(index_filename)
            local_foldernames.append(stream.local)
            if not num_stream_samples:
                raise RuntimeError(f'Stream contains no samples: {index_filename}.')
            stream_per_shard += [stream_id] * len(stream_shards)
            self.shard_offset_per_stream[stream_id] = len(self.shards)
            self.shards_per_stream[stream_id] = len(stream_shards)
            self.sample_offset_per_stream[stream_id] = self.num_samples
            self.samples_per_stream[stream_id] = num_stream_samples
            self.shards += stream_shards
            self.num_samples += num_stream_samples

        self.stream_per_shard = np.array(stream_per_shard, np.int64)
        self.num_shards = len(self.shards)

        # Check that cache limit is possible.
        if self.cache_limit:
            if isinstance(self.cache_limit, str):
                self.cache_limit = bytes_to_int(self.cache_limit)
            min_cache_usage = sum((stream.get_index_size() for stream in streams))
            if self.cache_limit <= min_cache_usage:
                raise ValueError(f'Minimum cache usage ({min_cache_usage} bytes) is larger than ' +
                                 f'the cache limit ({self.cache_limit} bytes). Please raise ' +
                                 f'`cache_limit`.')

        for stream_idx, index_filename in enumerate(index_filenames):
            if indices_created[stream_idx] == 0:
                # Index file was downloaded from remote.
                try:
                    # Remove the index.json file.
                    os.remove(index_filename)
                except FileNotFoundError:
                    pass
            elif indices_created[stream_idx] == 1:
                # Index file was local. Don't delete.
                pass
            else:
                # Directory and index file were created. Delete both.
                shutil.rmtree(local_foldernames[stream_idx])

        # Build the shard index (for partitioning and mapping samples to shards).
        self.samples_per_shard = np.array([shard.samples for shard in self.shards], np.int64)
        self.sample_offset_per_shard = self.samples_per_shard.cumsum() - self.samples_per_shard
        self.spanner = SimulationSpanner(self.samples_per_shard)

        # Also keep track of the raw and compressed sizes of each shard, indexed by shard_id.
        self.raw_shard_sizes = np.array([shard.get_raw_size() for shard in self.shards], np.int64)
        self.zip_shard_sizes = np.array([shard.get_zip_size() or 0 for shard in self.shards],
                                        np.int64)

        logger.info(f' Total number of shards: {self.num_shards}')
        logger.info(f' Average number of samples per shard: {self.num_samples / self.num_shards}')
        logger.info(f' Average raw shard size (bytes): {np.mean(self.raw_shard_sizes)}')
        logger.info(f' Average zip shard size (bytes): {np.mean(self.zip_shard_sizes)}')

        # Now that we know the number of underlying samples of each stream, derive each stream's
        # true proportion/repeat/choose, as well as the total epoch size.
        self.epoch_size = Stream.apply_weights(self.streams, self.samples_per_stream,
                                               epoch_size_value, self.shuffle_seed)

        # Length (__len__) is the resampled epoch size divided over the number of devices.
        self.length = ceil(self.epoch_size / self.world.num_ranks)

        t1 = time.time()
        self.instantiation_time = t1 - t0

        logger.info(' SimulationDataset created successfully.')

    def get_sample_partition(self, epoch: int, sample_in_epoch: int) -> NDArray:
        """Get the dataset's partition of this epoch's sample space.

        Args:
            epoch (int): Which epoch it is.
            sample_in_epoch (int): Where we are in the epoch.

        Returns:
            NDArray[np.int64]: Our partition of the epoch.
        """
        return generate_work(self.batching_method, self, self.world, epoch, sample_in_epoch)

    def get_samples_per_node(self, epoch: int, sample_in_epoch: int) -> NDArray:
        """Get the dataset's number of samples per node, worker, device.

        Args:
            epoch (int): Which epoch it is.
            sample_in_epoch (int): Where we are in the epoch.

        Returns:
            NDArray[np.int64]: The dataset's samples per node, worker, device.
        """
        partition = generate_work(self.batching_method, self, self.world, epoch, sample_in_epoch)
        # Modify partition to be in traversal order, per node, device, and worker.
        return partition.reshape(self.nodes, self.devices, self.workers, -1)

    def get_spanner(self) -> Spanner:
        """Get the dataset's spanner object, which does global sample id indexing.

        Returns:
            Spanner: The dataset's spanner object.
        """
        return self.spanner

    def get_raw_shard_sizes(self) -> NDArray[np.int64]:
        """Get the dataset's raw shard sizes.

        Returns:
            NDArray[np.int64]: The dataset's raw shard sizes.
        """
        return self.raw_shard_sizes

    def get_zip_shard_sizes(self) -> NDArray[np.int64]:
        """Get the dataset's zip shard sizes.

        Returns:
            NDArray[np.int64]: The dataset's zip shard sizes.
        """
        return self.zip_shard_sizes

    def get_nodes(self) -> int:
        """Get the dataset's number of nodes.

        Returns:
            int: The dataset's number of nodes.
        """
        return self.nodes

    def get_devices(self) -> int:
        """Get the dataset's number of devices.

        Returns:
            int: The dataset's number of devices.
        """
        return self.devices

    def get_workers(self) -> int:
        """Get the dataset's number of workers.

        Returns:
            int: The dataset's number of workers.
        """
        return self.workers

    def get_num_canonical_nodes(self) -> int:
        """Get the dataset's number of canonical nodes.

        Returns:
            int: The dataset's number of canonical nodes.
        """
        if not isinstance(self.num_canonical_nodes, int):
            raise TypeError(f'`self.num_canonical_nodes` must be an int. ' +
                            f'Got {type(self.num_canonical_nodes)} instead.')
        return self.num_canonical_nodes

    def get_batch_size(self) -> int:
        """Get the dataset's batch size.

        Returns:
            int: The dataset's batch size.
        """
        if not isinstance(self.batch_size, int):
            raise TypeError(f'`self.batch_size` must be an int. ' +
                            f'Got {type(self.batch_size)} instead.')
        return self.batch_size

    def get_num_shards(self) -> int:
        """Get the dataset's number of shards.

        Returns:
            int: The dataset's number of shards.
        """
        return self.num_shards

    def get_avg_samples_per_shard(self) -> int:
        """Get the dataset's average number of samples per shard.

        Returns:
            int: The dataset's average number of samples per shard.
        """
        return round(self.num_samples / self.num_shards)

    def get_predownload(self) -> int:
        """Get the dataset's predownload.

        Returns:
            int: The dataset's predownload.
        """
        if not isinstance(self.predownload, int):
            raise TypeError(f'`self.predownload` must be an int. ' +
                            f'Got {type(self.predownload)} instead.')
        return self.predownload

    def get_cache_limit(self) -> Optional[int]:
        """Get the dataset's cache limit.

        Returns:
            Optional[int]: The dataset's cache limit.
        """
        if isinstance(self.cache_limit, str):
            self.cache_limit = bytes_to_int(self.cache_limit)
        return self.cache_limit

    def get_instantiation_time(self) -> float:
        """Get the dataset's instantiation time.

        Returns:
            float: The dataset's instantiation time.
        """
        return self.instantiation_time

    def get_num_batches(self) -> int:
        """Get the dataset's number of batches.

        Returns:
            int: The dataset's number of batches.
        """
        if self.batch_size is None:
            raise ValueError(f'Cannot get number of batches without `batch size`, had ' +
                             f'`batch_size` of `None`')
        return self.epoch_size // (self.batch_size * self.devices * self.nodes)

    def get_stream_info(self) -> dict:
        """Get the dataset's stream info.

        Returns:
            dict: The dataset's stream info.
        """
        return self.stream_info

    def get_shuffle(self) -> bool:
        """Get the dataset's shuffle.

        Returns:
            bool: The dataset's shuffle.
        """
        return self.shuffle

    def get_shuffle_algo(self) -> str:
        """Get the dataset's shuffle algorithm.

        Returns:
            str: The dataset's shuffle algorithm.
        """
        return self.shuffle_algo

    def get_shuffle_seed(self) -> int:
        """Get the dataset's shuffle seed.

        Returns:
            int: The dataset's shuffle seed.
        """
        return self.shuffle_seed

    def get_shuffle_block_size(self) -> int:
        """Get the dataset's shuffle block size.

        Returns:
            int: The dataset's shuffle block size.
        """
        if not isinstance(self.shuffle_block_size, int):
            raise TypeError(f'`self.shuffle_block_size` must be an int. ' +
                            f'Got {type(self.shuffle_block_size)} instead.')
        return self.shuffle_block_size

    def get_epoch_size(self) -> int:
        """Get the dataset's epoch size.

        Returns:
            int: The dataset's epoch size.
        """
        return self.epoch_size

    def get_sampling_method(self) -> str:
        """Get the dataset's sampling method.

        Returns:
            str: The dataset's sampling method.
        """
        return self.sampling_method

    def get_sampling_granularity(self) -> int:
        """Get the dataset's sampling granularity.

        Returns:
            int: The dataset's sampling granularity.
        """
        return self.sampling_granularity

    def get_batching_method(self) -> str:
        """Get the dataset's batching method.

        Returns:
            str: The dataset's batching method.
        """
        return self.batching_method
