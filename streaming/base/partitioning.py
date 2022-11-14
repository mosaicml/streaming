# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from math import ceil
from typing import List, Optional, Tuple

import numpy as np


def _get_partition(global_device: int,
                   global_num_devices: int,
                   device_worker: int,
                   device_num_workers: int,
                   dataset_size: int,
                   batch_size: Optional[int] = None) -> Tuple[int, int]:
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

    Args:
        global_device (int): Rank.
        global_num_devices (int): Num ranks.
        device_worker (int): Worker of rank.
        device_num_workers (int): Workers per rank.
        dataset_size (int): Dataset size.
        batch_size (int, optional): Batch size.

    Returns:
        Partition: This worker's partition of the dataset.
    """

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

    device_min_id, _, device_samples = _get_min_max_size(0, dataset_size, global_device,
                                                         global_num_devices)

    # Some devices may have 1 fewer sample, so repeat some samples at boundaries
    expected_device_samples = ceil(dataset_size / global_num_devices)
    if device_samples < expected_device_samples:
        if device_samples != expected_device_samples - 1:
            raise RuntimeError('Found device partition with incorrect # samples')
        device_min_id -= 1
        device_samples += 1

    if not batch_size:
        worker_min_id, worker_max_id, _ = _get_min_max_size(device_min_id, device_samples,
                                                            device_worker, device_num_workers)
        return worker_min_id, worker_max_id

    device_batches = ceil(device_samples / batch_size)
    samples_missing = device_batches * batch_size - device_samples

    # Determine which batches this worker is responsible for
    worker_min_batch_id, worker_max_batch_id, _ = _get_min_max_size(0, device_batches,
                                                                    device_worker,
                                                                    device_num_workers)

    # The last device_worker to be read from will be the one with the incomplete batch.
    # This is done to match PyTorch DataLoader's round-robin scheduling of workers.
    # All device_workers must be careful to account for the missing samples offset by the
    # incomplete batch.
    incomplete_device_worker = (device_batches + device_num_workers - 1) % device_num_workers
    min_id_offset = 0 if device_worker <= incomplete_device_worker else samples_missing
    max_id_offset = 0 if device_worker < incomplete_device_worker else samples_missing

    worker_min_id = device_min_id + worker_min_batch_id * batch_size - min_id_offset
    worker_max_id = device_min_id + (worker_max_batch_id + 1) * batch_size - max_id_offset - 1

    # Adjustment for last partition.
    if dataset_size == worker_max_id:
        if worker_min_id:
            worker_min_id -= 1
        worker_max_id -= 1
    elif dataset_size < worker_max_id:
        raise RuntimeError('Partitions were calculated incorrectly')

    return worker_min_id, worker_max_id


def get_partitions(num_ranks: int,
                   workers_per_rank: int,
                   dataset_size: int,
                   batch_size: Optional[int] = None) -> List[Tuple[int, int]]:
    """Get the sample ID spans for each worker.

    Args:
        num_ranks (int): World size.
        workers_per_rank (int): Workers each rank.
        dataset_size (int): Number of samples in the dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        List[Tuple[int, int]]: Span per worker.
    """
    spans = []
    for rank in range(num_ranks):
        for worker_of_rank in range(workers_per_rank):
            span = _get_partition(rank, num_ranks, worker_of_rank, workers_per_rank, dataset_size,
                                  batch_size)
            spans.append(span)
    return spans
