# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Partitions the sample space to nodes, devices, and workers."""

import numpy as np


def get_partitions(num_samples: int,
                   num_canonical_nodes: int,
                   num_physical_nodes: int,
                   ranks_per_node: int,
                   workers_per_rank: int,
                   drop_first: int = 0):
    """Partition the given number of samples to nodes, devices, and workers.

    Args:
        num_samples (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes x ranks x workers x samples).
    """
    # Divide the full dataset sample range into sample range per canonical node.
    xx = []
    for i in range(num_canonical_nodes):
        a = i * num_samples // num_canonical_nodes
        z = (i + 1) * num_samples // num_canonical_nodes
        x = np.arange(a, z)
        xx.append(x)

    # Make the spans equal length by repeating the last sample if too short.
    # -> Shape: (canonical nodes x samples).
    max_len = max(map(len, xx))
    for i, x in enumerate(xx):
        if len(x) < max_len:
            last = np.array([x[-1]])
            xx[i] = np.concatenate([x, last])
    x = np.stack(xx)

    # If there are more physical than canonical nodes, interleave canonical onto physical nodes.
    # -> Shape: (canonical nodes x samples).
    if num_canonical_nodes < num_physical_nodes:
        assert not num_physical_nodes % num_canonical_nodes
        ratio = num_physical_nodes // num_canonical_nodes
        too_many = x.shape[1] % ratio
        if too_many:
            too_few = ratio - too_many
            last = x[:, -ratio - too_few + 1:-ratio + 1]
            x = np.concatenate([x, last], 1)
        x = x.reshape(ratio, -1, num_canonical_nodes)
        x = x.transpose(2, 1, 0)
        x = x.reshape(num_canonical_nodes, -1)

    # Drop samples that have already been seen and reshape.
    # -> Shape: (physical nodes x samples).
    x = x.transpose()
    x = x.flatten()
    x = x[drop_first:]
    x = x.reshape(-1, num_physical_nodes)
    x = x.transpose()

    # Interleave the node sample ranges over each node's devices, padding by repeating last sample.
    # -> Shape: (physical nodes x samples x devices).
    too_many = x.shape[1] % ranks_per_node
    if too_many:
        too_few = ranks_per_node - too_many
        last = x[:, -ranks_per_node - too_few + 1:-ranks_per_node + 1]
        x = np.concatenate([x, last], 1)
    x = x.reshape(num_physical_nodes, -1, ranks_per_node)

    # Interleave each device's samples across its workers, padding with -1.
    # -> Shape: (physical nodes x samples x workers x devices).
    too_many = x.shape[1] % workers_per_rank
    if too_many:
        too_few = workers_per_rank - too_many
        last = np.full((num_physical_nodes, too_few, ranks_per_node), -1, np.int64)
        x = np.concatenate([x, last], 1)
    x = x.reshape(num_physical_nodes, -1, workers_per_rank, ranks_per_node)

    # -> Shape: (physical nodes x devices x workers x samples).
    return x.transpose(0, 3, 2, 1)
