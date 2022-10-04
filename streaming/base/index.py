# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Helper methods to get the shard attributes."""

from math import ceil
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import get_worker_info

from streaming.base import distributed as dist


def get_index_basename() -> str:
    """Get the canonical index file basename.

    Returns:
        str: Index basename.
    """
    return 'index.json'


class Partition(object):
    """A worker's partition of the dataset.

    Args:
        shards (List[int]): The shards that this partition overlaps.
        shards_to_download (List[int]): The shards that this worker should download (subset of
            ``shards``).
        min_id (int): The lowest sample ID of this partition.
        max_id (int): The highest sample ID of this partition.
    """

    def __init__(self, shards: List[int], shards_to_download: List[int], min_sample_id: int,
                 max_sample_id: int) -> None:
        self.shards = shards
        self.shards_to_download = shards_to_download
        self.min_sample_id = min_sample_id
        self.max_sample_id = max_sample_id


class Index(object):
    """An index of sample ranges (corresponding to shards).

    Enables (a) finding the shard for a given sample, (b) getting the per-device dataset size, and
    (c) getting this device/worker's sample range of the dataset.
    """

    def __init__(self, samples_per_shard: List[int], batch_size: Optional[int] = None) -> None:
        self.samples_per_shard = samples_per_shard
        self.batch_size = batch_size

        self.total_samples = sum(samples_per_shard)
        self.shard_offsets = np.array([0] + samples_per_shard).cumsum().tolist()

        # Make a lookup table of sample to shard, stored in the form of equal-sized spans of sample
        # IDs that map to at most two adjacent shards, keeping the dividing sample ID.
        if samples_per_shard[:-1]:
            self.slot_size = min(samples_per_shard[:-1])
        else:
            self.slot_size = samples_per_shard[-1]
        self.slot_size = self.slot_size or 1  # For the edge case of empty shards.
        self.num_slots = (self.total_samples + self.slot_size - 1) // self.slot_size
        shard_ends = np.array(samples_per_shard).cumsum()
        shard = 0
        slots = []
        for slot in range(self.num_slots):
            slot_end = (slot + 1) * self.slot_size
            if shard_ends[shard] < slot_end:
                div = shard_ends[shard]
                slots.append((shard, div))
                shard += 1
            else:
                div = slot_end
                slots.append((shard, div))
        self.slots = np.array(slots)

    def find_sample(self, idx: int) -> Tuple[int, int]:
        """Get the shard and offset where a sample will be found.

        Args:
            idx (int): Global sample index.

        Returns:
            Tuple[int, int]: Shard and sample index within that shard.
        """
        slot_idx = min(idx // self.slot_size, len(self.slots) - 1)
        shard, div = self.slots[slot_idx]
        if div <= idx:
            shard += 1
        offset = idx - self.shard_offsets[shard]
        return shard, offset

    def get_samples_per_device(self) -> int:
        """Get the per-device dataset size (i.e., IterableDataset.__len__).

        Returns:
            int: Per-device dataset size.
        """
        return ceil(self.total_samples / dist.get_world_size())

    def get_partition(self) -> Partition:
        """Get the shards and sample range of this device/worker's partition.

        When ``batch_size`` is provided, worker indices will be constructed so that there is at
        most one incomplete batch at the end of each epoch. For example, if the DataLoader is
        reading over::

            samples: [0, 1, 2, 3, 4, 5, 6, 7]
            num_workers: 3
            batch_size: 2
            drop_last: True

        but ``batch_size`` is not hinted to the StreamingDataset ahead of time, then the samples
        will by default be assigned like::

            worker 0: [0, 1, 2]
            worker 1: [3, 4, 5]
            worker 2: [6, 7]

        and will be read as batches like (with samples [2] and [5] dropped as incomplete)::

            batch 0: [0, 1]
            batch 1: [3, 4]
            batch 2: [6, 7]

        The above is suboptimal because we could have dropped no samples. So when ``batch_size`` is
        provided as a hint, we assign samples like this::

            worker 0: [0, 1, 2, 3]
            worker 1: [4, 5]
            worker 2: [6, 7]

        which will be read as batches like::

            batch 0: [0, 1]
            batch 1: [4, 5]
            batch 2: [6, 7]
            batch 3: [2, 3]

        Returns:
            Partition: This worker's partition of the dataset.
        """
        global_device = dist.get_global_rank()
        global_num_devices = dist.get_world_size()
        node_device = dist.get_local_rank()
        node_num_devices = dist.get_local_world_size()

        worker_info = get_worker_info()
        if worker_info:
            device_worker = worker_info.id
            device_num_workers = worker_info.num_workers
        else:
            device_worker = 0
            device_num_workers = 1
        node_worker = node_device * device_num_workers + device_worker
        node_num_workers = node_num_devices * device_num_workers

        # Splits a range (start, start+total) into num_parts such that:
        # each part spans a continguous range [part_min_id, part_max_id]
        # each part_i starts immediately from where the previous part_[i-1] stopped
        # all parts have the same number of items,
        # except the first K parts may have exactly 1 more item
        def _get_min_max_size(start: int, total: int, part: int, num_parts: int):
            sizes = [ceil((total - p) / num_parts) for p in range(num_parts)]
            min_ids = np.cumsum([0] + sizes)
            part_min_id = start + min_ids[part]
            part_max_id = start + min_ids[part + 1] - 1
            part_size = sizes[part]
            return part_min_id, part_max_id, part_size

        device_min_id, _, device_samples = _get_min_max_size(0, self.total_samples, global_device,
                                                             global_num_devices)

        # Some devices may have 1 fewer sample, so repeat some samples at boundaries
        expected_device_samples = ceil(self.total_samples / global_num_devices)
        if device_samples < expected_device_samples:
            if device_samples != expected_device_samples - 1:
                raise RuntimeError('Found device partition with incorrect # samples')
            device_min_id -= 1
            device_samples += 1

        if not self.batch_size:
            worker_min_id, worker_max_id, _ = _get_min_max_size(device_min_id, device_samples,
                                                                device_worker, device_num_workers)
        else:
            device_batches = ceil(device_samples / self.batch_size)
            samples_missing = device_batches * self.batch_size - device_samples

            # Determine which batches this worker is responsible for
            worker_min_batch_id, worker_max_batch_id, _ = _get_min_max_size(
                0, device_batches, device_worker, device_num_workers)

            # The last device_worker to be read from will be the one with the incomplete batch.
            # This is done to match PyTorch DataLoader's round-robin scheduling of workers.
            # All device_workers must be careful to account for the missing samples offset by the
            # incomplete batch.
            incomplete_device_worker = \
                (device_batches + device_num_workers - 1) % device_num_workers
            min_id_offset = 0 if device_worker <= incomplete_device_worker else samples_missing
            max_id_offset = 0 if device_worker < incomplete_device_worker else samples_missing

            worker_min_id = device_min_id + worker_min_batch_id * self.batch_size - min_id_offset
            worker_max_id = \
                device_min_id + (worker_max_batch_id + 1) * self.batch_size - max_id_offset - 1

            # Adjustment for last partition.
            if self.total_samples == worker_max_id:
                if 0 < worker_min_id:
                    worker_min_id -= 1
                worker_max_id -= 1
            elif self.total_samples < worker_max_id:
                raise RuntimeError('Invalid partitioning')

        min_shard, _ = self.find_sample(worker_min_id)
        max_shard, _ = self.find_sample(worker_max_id)
        shards = list(range(min_shard, max_shard + 1))

        # Ensure that each shard only gets downloaded by 1 worker, so there are no race conditions.
        # To do this, we skip downloading the last shard (likely overlapped with the next worker)
        # unless:
        # - you are the last worker on your node (no files shared across nodes so you have to
        #   download it again!)
        # - you are downloading the last sample of the shard (no overlap with next worker)
        max_shard_next, _ = self.find_sample(worker_max_id + 1)
        if ((node_worker + 1 == node_num_workers) or
            (worker_max_id + 1 < self.total_samples and max_shard_next != max_shard)):
            shards_to_download = shards
        else:
            shards_to_download = shards[:-1]
        return Partition(shards, shards_to_download, worker_min_id, worker_max_id)
