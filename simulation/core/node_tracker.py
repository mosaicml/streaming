# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Class for tracking node information during simulation."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.last_used_ordered_set import LastUsedOrderedSet
from core.utils import remove_padded_samples
from numpy.typing import NDArray
from sortedcollections import OrderedSet
from streaming.base.spanner import Spanner
from typing import Optional, Tuple

class NodeTracker():

    def __init__(self, workers: int, devices: int, predownload: int,
                 device_batch_size: int, cache_limit: Optional[int] = None):
        """Tracker for node information during simulation.

        Args:
            node_id (int): The node ID.
            workers (int): The number of workers.
            devices (int): The number of devices.
            predownload (int): The number of samples to predownload.
            device_batch_size (int): The device batch size.
            sample_to_shard (Spanner): The mapping from samples to shards.
            cache_limit (Optional[int]): The cache limit for the node. Defaults to None.
        """
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

        # Use the set_epoch_samples method every epoch to set the node's samples.
        self.samples = None
    
    def initialize_worker_downloads(self, sample_to_shard: Spanner):
        """Initialize the worker downloads."""
        # For downloads, we round-robin over devices first, then workers.
        if self.samples is None:
            raise ValueError("Must set samples before initializing worker downloads.")
        else:
            for worker in range(self.workers):
                for device in range(self.devices):
                    download_samples = remove_padded_samples(self.samples[device, worker, 
                                                                            :self.predownload])
                    # Get the shards these samples correspond to, maintaining access order.
                    download_shards = OrderedSet([sample_to_shard[sample] 
                                                    for sample in download_samples])
                    self.worker_downloads.append(download_shards)

    def set_shards_used(self, shards: set,
                        shard_access_ends: NDArray,
                        step_num: int):
        """Set a set of shards as used.

        Args:
            shards (set): The shards to set as used.
            shard_access_ends (NDArray): The shard access end steps.
            step_num (int): The current step number.
        """
        for shard in shards:
            self.shards.setuse(shard)
            # For any shard access, we are accessing the shard so we need the shard until
            # at least the next step begins. Adding 0.5 ensures that we evict shards
            # after they are used for the last time, but before they are replaced by 
            # new downloads in the next step.
            shard_access_ends[shard] = step_num + 0.5
    
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
    
    def evict_until_satisfied(self, incoming_shard_size: int, raw_shard_sizes: NDArray):
        """Evict shards until the node has enough space to download the incoming shard.

        Args:
            incoming_shard_size (int): The size of the incoming shard.
            raw_shard_sizes (NDArray): The raw shard sizes.
        """
        while self.cache_usage + incoming_shard_size > self.cache_limit:
            evicted_shard = self.evict_shard()
            self.cache_usage -= raw_shard_sizes[evicted_shard]

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
            return self.worker_downloads[index]
        elif worker is not None and device is not None:
            return self.worker_downloads[worker * self.devices + device]
        else:
            raise ValueError("Must specify either index, or worker and device.")
    
    def get_current_batch_shards(self, worker: int, 
                                 worker_sample_index: int,
                                 sample_to_shard: Spanner) -> Tuple[set, set]:
        """Get this node's shards for the current batch.

        Args:
            worker (int): The worker.
            worker_sample_index (int): The worker sample index.
            sample_to_shard (Spanner): The mapping from samples to shards.
        Returns:
            Tuple[set, set]: shard ids needed by node, shard ids present in node.
        """
        batch_samples = remove_padded_samples(self.samples[:, worker, 
                                     worker_sample_index:
                                     worker_sample_index + self.device_batch_size].flatten())
        batch_shards = set([sample_to_shard[sample] for sample in batch_samples])
        shards_needed = batch_shards.difference(self.shards.keys())
        shards_present = batch_shards.difference(shards_needed)
        return shards_needed, shards_present
    
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
    
    def update_worker_predownloads(self, worker: int,
                                       worker_sample_index: int,
                                       sample_to_shard: Spanner):
        """Get the worker predownload samples for a worker and device.

        Args:
            worker (int): The current batch worker index.
            worker_sample_index (int): The worker sample index.
            sample_to_shard (Spanner): The mapping from samples to shards.
        """
        for device in range(self.devices):
            new_download_samples = remove_padded_samples(self.samples[device, worker,
                                                worker_sample_index + self.predownload:
                                                worker_sample_index + self.device_batch_size + 
                                                self.predownload])
            
            # Want to maintain the shard access order.
            new_download_shards = OrderedSet([sample_to_shard[sample] 
                                            for sample in new_download_samples])
            
            worker_downloads = self.get_worker_download(worker=worker, device=device)

            # Add in new shards to the worker's shard downloads only if the node does not yet have it.
            for shard in new_download_shards:
                if shard not in self.shards:
                    worker_downloads.add(shard)

    