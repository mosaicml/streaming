# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Evicting soon-to-be cold shards at exact times."""

from typing import List

import numpy as np
from numpy.typing import NDArray


def get_evictions_per_worker(node_shard_ids: NDArray[np.int64],
                             shard_ttls: NDArray[np.float64]) -> List[NDArray[np.int64]]:
    """Calculate shard evictions across workers given node shard IDs and shard TTLs.

    Args:
        node_shard_ids (NDArray[np.int64]): This node this epoch's shard ID tensor (ranks per node,
            workers per rank, batches per worker, batch size).
        shard_ttls (NDArray[np.float64]): TTL (time to live) per shard, as a fraction of the number
            of timesteps this epoch. Set to 1 if cache shards forever.

    Returns:
        List[NDArray[np.int64]]: Evictions divided across workers.
    """
    ranks_per_node, workers_per_rank, batches_per_worker, batch_size = node_shard_ids.shape

    # Reshape shard IDs to a matrix of (timestep, ranks per node), where each row happens all at
    # the same time and rows are in temporal order.
    shard_ids = node_shard_ids.transpose(2, 1, 3, 0)
    shard_ids = shard_ids.reshape(-1, ranks_per_node)

    # Then rearrange the rows to make the shard IDs occur in ascending order, so that we can skip
    # over duplicate shards easily.
    shard_ids.sort(0)

    # Get shard evictions per timestep over all workers.
    last_seen = np.zeros(len(shard_ttls), np.int64) - 1
    num_timesteps = batches_per_worker * workers_per_rank * batch_size
    evictions = []
    for timestep in range(num_timesteps):
        prev_shard_id = -1
        for shard_id in shard_ids[timestep]:
            if shard_id == prev_shard_id:
                continue
            prev_shard_id = shard_id
            if last_seen[shard_id] != -1:
                prev_timestep = last_seen[shard_id]
                interval_frac = (timestep - prev_timestep) / num_timesteps
                if shard_ttls[shard_id] < interval_frac:
                    evictions.append((prev_timestep, shard_id))
            last_seen[shard_id] = timestep

    # Get the final eviction for each shard that was used and should be evicted.
    for shard_id, shard_ttl in enumerate(shard_ttls):
        prev_timestep = last_seen[shard_id]
        if prev_timestep != -1 and shard_ttl < 1:
            evictions.append((prev_timestep, shard_id))

    # Sort eviction pairs of (timestep, shard ID) into an array.
    evictions.sort()
    evictions = np.array(evictions, np.int64)

    # Assign each eviction pair to the appropriate worker.
    index = 0
    timestep = 0
    evictions_per_worker = [[] for _ in range(workers_per_rank)]
    for _ in range(batches_per_worker):
        for worker in range(workers_per_rank):
            for _ in range(batch_size):
                while index < len(evictions) and evictions[index][0] == timestep:
                    shard_id = evictions[index][1]
                    evictions_per_worker[worker].append((timestep, shard_id))
                    index += 1
                timestep += 1

    # Convert lists to arrays.
    return list(map(np.array, evictions_per_worker))
