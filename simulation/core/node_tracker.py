# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Class for tracking node information during simulation."""

from typing import Optional, Tuple

import numpy as np
from core.last_used_ordered_set import LastUsedOrderedSet
from core.utils import remove_padded_samples
from numpy.typing import NDArray
from sortedcollections import OrderedSet

from streaming.base.spanner import Spanner


class NodeTracker():
    """Tracker for node information during simulation.

    Args:
        workers (int): The number of workers.
        devices (int): The number of devices.
        predownload (int): The number of samples to predownload.
        device_batch_size (int): The device batch size.
        total_shards (int): Total number of shards in the dataset.
        cache_limit (Optional[int]): The cache limit for the node. Defaults to ``None``.
    """

    def __init__(self,
                 workers: int,
                 devices: int,
                 predownload: int,
                 device_batch_size: int,
                 total_shards: int,
                 cache_limit: Optional[int] = None):

        self.shards = LastUsedOrderedSet()
        self.all_shards = set()
        self.cache_usage = 0
        self.partial_shard_bytes = 0
        self.worker_download_index = 0
        self.devices = devices
        self.workers = workers
        self.total_workers = workers * devices
        self.device_batch_size = device_batch_size
        self.predownload = predownload
        self.cache_limit = cache_limit
        self.worker_downloads = []
        self.shard_access_starts = np.full(total_shards, -1)
        self.shard_access_ends = np.full(total_shards, -1)

        # Use the set_epoch_samples method every epoch to set the node's samples.
        self.samples = None

    def initialize_worker_downloads(self, sample_to_shard: Spanner):
        """Initialize worker downloads, making shards in the predownload sample range available.

        Args:
            sample_to_shard (Spanner): The mapping from samples to shards.
        """
        # For downloads, we round-robin over devices first, then workers.
        if self.samples is not None:
            for worker in range(self.workers):
                for device in range(self.devices):
                    download_samples = remove_padded_samples(
                        self.samples[device, worker, :self.predownload])
                    # Get the shards these samples correspond to, maintaining access order.
                    download_shards = OrderedSet(
                        [sample_to_shard[sample] for sample in download_samples])
                    self.worker_downloads.append(download_shards)
        else:
            raise AttributeError('Must set node samples before accessing them.')

    def set_shards_used(self, shards: set, step_num: int):
        """Mark a set of shards as recently used.

        Args:
            shards (set): The shards to set as used.
            step_num (int): The current step number.
        """
        for shard in shards:
            self.shards.setuse(shard)
            # For any shard access, we are accessing the shard so we need the shard until
            # at least the next step begins. Adding 0.5 ensures that we evict shards
            # after they are used for the last time, but before they are replaced by
            # new downloads in the next step.
            self.shard_access_ends[shard] = step_num + 0.5

    def add_shard(self, shard: int, used: bool = True):
        """Add a shard to the node.

        Args:
            shard (int): The shard to add.
            used (bool): Whether the shard is used when added.
        """
        self.shards.setitem(shard, used)
        self.all_shards.add(shard)

    def get_all_shards(self):
        """Get all the shards in the node."""
        return self.all_shards

    def evict_shard(self) -> int:
        """Evict a shard.

        Returns:
            int: The evicted shard.
        """
        evicted_shard = self.shards.popLRU()
        return evicted_shard

    def evict_until_satisfied(self, incoming_shard_size: int, raw_shard_sizes: NDArray[np.int64]):
        """Evict shards until the node has enough space to download the incoming shard.

        Args:
            incoming_shard_size (int): The size of the incoming shard.
            raw_shard_sizes (NDArray[np.int64]): The raw shard sizes.
        """
        # We evict shards until the incoming shard fits into the node's cache.
        if self.cache_limit is not None:
            while self.cache_usage + incoming_shard_size > self.cache_limit:
                evicted_shard = self.evict_shard()
                self.cache_usage -= int(raw_shard_sizes[evicted_shard])

    def increment_worker_download_index(self):
        """Increment the worker download index."""
        self.worker_download_index = (self.worker_download_index + 1) % \
            (self.workers * self.devices)

    def get_worker_download(self,
                            worker: Optional[int] = None,
                            device: Optional[int] = None,
                            index: Optional[int] = None) -> OrderedSet:
        """Get the shard downloads for a worker on a specific device.

        Args:
            worker (Optional[int]): The worker index.
            device (Optional[int]): The device index the worker is on.
            index (Optional[int]): The index of the worker download for direct access.

        Returns:
            OrderedSet: The shard downloads, in order, for this worker.
        """
        if index is not None:
            # Directly access worker_downloads through an index.
            return self.worker_downloads[index]
        elif worker is not None and device is not None:
            # Access worker_downloads through worker and device indices.
            return self.worker_downloads[worker * self.devices + device]
        else:
            raise ValueError('Must specify either index, or worker and device.')

    def get_current_batch_shards(self, worker: int, worker_sample_index: int,
                                 sample_to_shard: Spanner) -> Tuple[set, set]:
        """Get this node's shards for the current batch.

        Args:
            worker (int): The worker.
            worker_sample_index (int): The worker sample index.
            sample_to_shard (Spanner): The mapping from samples to shards.

        Returns:
            Tuple[set, set]: shard ids needed by node, shard ids present in node.
        """
        if self.samples is not None:
            batch_samples = remove_padded_samples(
                self.samples[:, worker, worker_sample_index:worker_sample_index +
                             self.device_batch_size].flatten())
            batch_shards = {sample_to_shard[sample] for sample in batch_samples}
            shards_needed = batch_shards.difference(self.shards.keys())
            shards_present = batch_shards.difference(shards_needed)
            return shards_needed, shards_present
        else:
            raise AttributeError('Must set node samples before accessing them.')

    def get_next_worker_with_downloads(self) -> Optional[OrderedSet]:
        """Get the next worker with samples to download.

        Returns:
            Optional[OrderedSet]: The next worker's shard downloads, or None if no workers have
                samples to download.
        """
        empty_download_counter = 0
        worker_download = self.get_worker_download(index=self.worker_download_index)
        while len(worker_download) == 0:
            empty_download_counter += 1
            self.increment_worker_download_index()
            worker_download = self.get_worker_download(index=self.worker_download_index)

            # No workers in the node have samples to download.
            if empty_download_counter >= self.total_workers:
                return None

        return worker_download

    def update_worker_predownloads(self, worker: int, worker_sample_index: int,
                                   sample_to_shard: Spanner):
        """Update the worker predownload samples for a worker and device.

        Args:
            worker (int): The current batch worker index.
            worker_sample_index (int): The worker sample index.
            sample_to_shard (Spanner): The mapping from samples to shards.
        """
        if self.samples is not None:
            for device in range(self.devices):
                # Retrieve new samples that are now within predownload range of the worker.
                new_download_samples = remove_padded_samples(
                    self.samples[device, worker,
                                 worker_sample_index + self.predownload:worker_sample_index +
                                 self.device_batch_size + self.predownload])

                # Want to maintain the shard access order.
                new_download_shards = OrderedSet(
                    [sample_to_shard[sample] for sample in new_download_samples])

                worker_downloads = self.get_worker_download(worker=worker, device=device)

                # Add in new shards to the worker's shard downloads only if node does not yet have it.
                for shard in new_download_shards:
                    if shard not in self.shards:
                        worker_downloads.add(shard)
        else:
            raise AttributeError('Must set node samples before accessing them.')
