# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Near replica of StreamingDataset for simulation purposes."""

import logging
import os
import shutil
import time
import warnings
from math import ceil
from multiprocessing import Pool
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from core.sim_spanner import SimulationSpanner
from core.sim_world import SimulationWorld
from numpy.typing import NDArray

from streaming import Stream, StreamingDataset
from streaming.batching import generate_work
from streaming.format import get_index_basename
from streaming.format.base.phaser import Phaser
from streaming.spanner import Spanner
from streaming.util.shorthand import normalize_bytes, normalize_count

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SimulationDataset(StreamingDataset):
    """Near replica of StreamingDataset for simulation purposes.

    Args:
        nodes (int): Number of nodes.
        devices (int): Number of devices.
        workers (int): Number of workers.
        streams (Sequence[Stream], optional): One or more streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        allow_schema_mismatch (bool): If ``True``, continue if sample columns mismatch across
            shards, streams, or the whole dataset. If ``False``, raises if columns mismatch.
            Defaults to ``False``.
        allow_unsafe_types (bool): If ``True``, continue if unsafe type(s) are encountered
            in shard(s). If ``False``, raises if unsafe type(s) encountered. Defaults to ``False``.
        allow_unchecked_resumption (bool): If ``True``, upon resume, accept and use shard
            file phases that we are unable to check the size/hash(es) of. If ``False``, upon
            resume, drop such files, to regenerate on the fly when needed. Defaults to ``True``.
        download_retry (int): Number of download re-attempts before raising an error. Defaults to
            ``2``.
        download_timeout (str | float): Time in seconds to wait for a file download to complete
            before raising an error. Streaming duration shorthand (e.g., ``1m23s``) is also
            accepted. Defaults to ``1m``.
        validate_hash (str, optional): Deprecated. See ``hash_algos``. Defaults to ``None``.
        keep_phases (str | Sequence[str] | Dict[str, bool] | Phaser, optional): After a phase
            transition of a shard file, do we keep the old form of the file or garbage collect it?
            Provided as one of: (1) ``None`` for defaults, (2) the single use case or phase to keep,
            (3) a sequence giving the use cases or phases to keep, (4) Phaser kwargs (a mapping of
            use case or phase to whether it must be kept, or (5) a Phaser object. All code paths
            result in a ``Phaser``. Defaults to ``None``.
        epoch_size (Union[str, int], optional): Number of samples to draw per epoch balanced
            across all streams. If ``None``, takes its value from the total number of underlying
            samples. Provide this field if you are weighting streams relatively to target a larger
            or smaller epoch size. Defaults to ``None``. Can also take in human-readable number
            abbreviations (e.g., ``"100k"``, ``"64M"``, ``"77b"``, etc). Defaults to ``None``.
        predownload (int, optional): Target number of samples to download per worker in advance
            of current sample. Workers will attempt to download ahead by this many samples during,
            but not before, training. Recommendation is to provide a value greater than per device
            batch size to ensure at-least per device batch size number of samples cached locally.
            If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[str, int], optional): Maximum size in bytes of this StreamingDataset's
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
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
    """

    def __init__(
        self,
        *,
        nodes: int,
        devices: int,
        workers: int,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        allow_schema_mismatch: bool = False,
        allow_unsafe_types: bool = False,
        allow_unchecked_resumption: bool = True,
        download_retry: int = 2,
        download_timeout: Union[str, float] = '2m',
        download_max_size: Optional[Union[str, int]] = '200mb',
        validate_hash: Optional[str] = None,
        keep_phases: Union[None, str, Sequence[str], Dict[str, bool], Phaser] = None,
        epoch_size: Optional[Union[str, int]] = None,
        predownload: Optional[int] = None,
        cache_limit: Optional[Union[str, int]] = None,
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
        **kwargs: Any,
    ) -> None:
        # Time how long it takes for StreamingDataset instantiation
        t0 = time.time()

        self.nodes = nodes
        self.devices = devices
        self.workers = workers

        # Global arguments (which do not live in Streams).
        self.predownload = predownload
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

        self.batch_size = batch_size or 1

        # Convert epoch size from string to int, if needed. Cannot be negative.
        epoch_size_value = normalize_count(epoch_size) if epoch_size else None

        # Initialize the Stream defaults and normalize to a list of Streams.
        kwargs.update(
            split=split,
            allow_schema_mismatch=allow_schema_mismatch,
            allow_unsafe_types=allow_unsafe_types,
            allow_unchecked_resumption=allow_unchecked_resumption,
            download_retry=download_retry,
            download_timeout=download_timeout,
            download_max_size=download_max_size,
            validate_hash=validate_hash,
            keep_phases=keep_phases,
        )
        if streams:
            for stream in streams:
                stream.apply_defaults(**kwargs)
        else:
            streams = Stream(remote=remote, local=local, **kwargs),

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
                filepath = os.path.join(stream.remote, stream.split or '', get_index_basename())
                indices_created.append(0)
            else:
                filepath = os.path.join(stream.local, stream.split or '', get_index_basename())
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
                'proportion': getattr(stream, 'proportion', None),
                'repeat': getattr(stream, 'repeat', None),
                'choose': getattr(stream, 'choose', None),
            }

        # Initialize the SimulationWorld, which tells us about nodes/devices/workers
        self.world = SimulationWorld(self.nodes, self.devices, self.workers)

        # Download each Stream's index in parallel from local rank 0.
        #
        # Parallelism is important because there could be a very large number of Streams, and we
        # expect equal performance as them having been concatenated into one Stream.
        if self.world.is_local_leader:
            pool = Pool()
            pool.imap_unordered(Stream.download_index, self.streams)
            pool.close()
        else:
            pool = None

        # All ranks then walk all Streams, for which they (1) wait for its index to become
        # downloaded, (2) load its shards, and (3) map streams <-> shards <-> samples.
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
            stream.await_index()
            stream_shards = stream.load_index()
            num_stream_samples = sum(map(len, stream_shards))
            index_filename = os.path.join(stream.local, stream.split or '', get_index_basename())
            index_filenames.append(index_filename)
            local_foldernames.append(stream.local)
            if not num_stream_samples:
                raise RuntimeError(f'Stream contains no samples: {stream.local_index_path}.')
            stream_per_shard += [stream_id] * len(stream_shards)
            self.shard_offset_per_stream[stream_id] = len(self.shards)
            self.shards_per_stream[stream_id] = len(stream_shards)
            self.sample_offset_per_stream[stream_id] = self.num_samples
            self.samples_per_stream[stream_id] = num_stream_samples
            self.shards += stream_shards
            self.num_samples += num_stream_samples
        self.stream_per_shard = np.array(stream_per_shard, np.int64)
        self.num_shards = len(self.shards)

        # Wait for the pool workers (stream index download processes) to finish.
        if pool is not None:
            pool.join()

        # Check that cache limit is possible.
        if cache_limit:
            self.cache_limit = normalize_bytes(cache_limit)
            min_cache_usage = sum((stream.index_size for stream in streams))
            if self.cache_limit <= min_cache_usage:
                raise ValueError(f'Minimum cache usage ({min_cache_usage} bytes) is larger than ' +
                                 f'the cache limit ({self.cache_limit} bytes). Please raise ' +
                                 f'`cache_limit`. Recommendation is to provide a `cache_limit` ' +
                                 f'as high as possible to avoid thrashing.')

            max_phase_size = 0
            for stream in self.streams:
                for shard in stream.shards:
                    for file in shard.files:
                        for phase in file.phases:
                            if not phase:
                                continue
                            if not phase.expected_size:
                                continue
                            if max_phase_size < phase.expected_size:
                                max_phase_size = phase.expected_size

            if self.cache_limit < 4 * max_phase_size:
                raise ValueError(f'Cache limit ({self.cache_limit} bytes) is too low. ' +
                                 f'Increase the `cache_limit` to at-least four times the ' +
                                 f'largest shard size ({max_phase_size} ' +
                                 f'bytes) which includes raw (decompressed) and zip ' +
                                 f'(compressed) file size. Recommendation is to provide a ' +
                                 f'`cache_limit` as high as possible to avoid thrashing.')
        else:
            self.cache_limit = None

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

        # Also keep track of the raw and compressed sizes of each shard.
        for shard in self.shards:
            if len(shard.files) > 1:
                raise ValueError(f'The Streaming Simulator currently only supports datasets ',
                                 f'in MDS format. Please make sure your datasets are in MDS.')
        self.raw_shard_sizes = np.array([shard.files[0].raw_phase.expected_size \
                                         for shard in self.shards],
                                        np.int64)
        self.zip_shard_sizes = np.array([shard.files[0].zip_phase.expected_size or 0 \
                                         if shard.files[0].zip_phase is not None \
                                         else 0 \
                                         for shard in self.shards],
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
