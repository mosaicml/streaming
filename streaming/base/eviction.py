# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Evicting soon-to-become cold shards at specific times."""

from multiprocessing.shared_memory import SharedMemory
from typing import Iterator, List, Sequence

import numpy as np
from numpy.typing import NDArray

from streaming.base.stream import Stream


def pack_evictions(lists: List[List[int]]) -> NDArray[np.int64]:
    """Convert from shard eviction list per timestep to packed numpy array.

    Args:
        lists (List[List[int]]): Shard eviction list per timestep.

    Returns:
        NDArray[np.int64]: Packed numpy array.
    """
    arr = [-1, len(lists)]
    for timestep, evictions in enumerate(lists):
        if not evictions:
            continue
        arr += [timestep, len(evictions)] + evictions
    arr[0] = len(arr)
    return np.array(arr)


def unpack_evictions(arr: NDArray[np.int64]) -> Iterator[List[int]]:
    """Convert from packed numpy array to shard eviction list per timestep.

    Args:
        arr (NDArray[np.int64]): Packed numpy array.

    Returns:
        Iterator[List[int]]: Shard eviction list per timestep.
    """
    size, num_timesteps = arr[:2]
    if size != len(arr):
        raise ValueError('Invalid packed evictions data')
    idx = 2
    for timestep in range(num_timesteps):
        if (size <= idx) or (timestep < arr[idx]):
            yield []
            continue
        count = arr[idx + 1]
        yield arr[idx + 2:idx + 2 + count].tolist()
        idx += 2 + count


def get_evictions(streams: Sequence[Stream], shards_per_stream: NDArray[np.int64],
                  samples_per_shard: NDArray[np.int64],
                  node_sample_ids: NDArray[np.int64]) -> List[NDArray[np.int64]]:
    """Calculate shard evictions given this node this epoch's sample ID tensor.

    Args:
        node_sample_ids (NDArray[np.int64]): Sample ID tensor of shape (ranks per node, workers
            per rank, batches per worker, batch size).

    Returns:
        List[NDArray[np.int64]]: Packed evictions per worker.
    """
    ranks_per_node, workers_per_rank, batches_per_worker, batch_size = node_sample_ids.shape

    # Convert sample IDs to shard IDs, handling -1s.
    shard_ids = np.arange(len(samples_per_shard))
    sample_to_shard = np.repeat(shard_ids, samples_per_shard)
    node_shard_ids = np.where(node_sample_ids != -1, sample_to_shard[node_sample_ids], -1)

    # Reshape shard IDs to a matrix of (timestep, ranks per node), where each row happens all at
    # the same time and rows are in temporal order.
    shard_ids = node_shard_ids.transpose(2, 1, 3, 0)
    shard_ids = shard_ids.reshape(-1, ranks_per_node)

    # Then rearrange the rows to make the shard IDs occur in ascending order, so that we can skip
    # over duplicate shards easily.
    shard_ids.sort(0)

    # Gather keep_raw_interval per shard for efficient lookup.
    int64_max = np.iinfo(np.int64).max
    keep_raw_intervals = np.zeros(len(streams), np.int64)
    for idx, stream in enumerate(streams):
        keep_raw_intervals[idx] = int64_max if stream.keep_raw else stream.keep_raw_interval
    keep_raw_intervals = np.repeat(keep_raw_intervals, shards_per_stream)

    # Get the list of shard evictions per timestep for this node for all workers.
    last_seen = np.zeros(len(samples_per_shard), np.int64) - 1
    num_timesteps = batches_per_worker * workers_per_rank * batch_size
    evictions = [[] for _ in range(num_timesteps)]
    for timestep in range(num_timesteps):
        prev_shard_id = -1
        for shard_id in shard_ids[timestep]:
            if shard_id == prev_shard_id:
                continue
            prev_shard_id = shard_id
            if last_seen[shard_id] != -1:
                prev_timestep = last_seen[shard_id]
                interval = timestep - prev_timestep
                if keep_raw_intervals[shard_id] < interval:
                    evictions[prev_timestep].append(shard_id)
            last_seen[shard_id] = timestep

    # Divide those evictions into each worker's piece. Shape: (workers per rank, batches per worker
    # * batch size).
    #
    # Note that the PyTorch dataloader cycles round robin through the workers, so the evictions
    # have to be split up across workers accordingly (we can't do them all from the local
    # leader, unfortunately).
    evictions_per_worker = [[] for _ in range(workers_per_rank)]
    idx = 0
    for _ in range(batches_per_worker):
        for worker in range(workers_per_rank):
            for _ in range(batch_size):
                evictions_per_worker[worker].append(evictions[idx])
                idx += 1

    # Pack lists of lists into flat arrays to better use memory.
    return list(map(pack_evictions, evictions_per_worker))


def share_evictions(prefix: str, evictions: List[NDArray[np.int64]]) -> List[SharedMemory]:
    """Put shard evictions data into shared memory.

    Args:
        prefix (str): Shared memory name prefix.
        evictions (List[NDArray[np.int64]]): Shard evictions data.

    Returns:
        List[SharedMemory]: Handle to shared memory object per worker.
    """
    shms = []
    for worker, arr in enumerate(evictions):
        if not worker:
            continue
        name = f'{prefix}_evict_{worker}'
        size = arr.size * np.int64().nbytes
        shm = SharedMemory(name, True, size)
        shm.buf[:] = arr.tobytes()
        shms.append(shm)
    return shms


def attach_evictions(prefix: str, worker_of_rank: int) -> NDArray[np.int64]:
    """Get shard evictions data from shared memory.

    Args:
        prefix (str): Shared memory name prefix.
        worker_of_rank (int): Worker of rank.

    Returns:
        NDArray[np.int64]: Shard eviction data.
    """
    name = f'{prefix}_evict_{worker_of_rank}'
    shm = SharedMemory(name, False, np.int64().nbytes)
    arr = np.ndarray(1, buffer=shm.buf, dtype=np.int64)
    size = arr[0]
    shm = SharedMemory(name, False, size * np.int64().nbytes)
    arr = np.ndarray(size, buffer=shm.buf, dtype=np.int64)
    return arr.copy()
