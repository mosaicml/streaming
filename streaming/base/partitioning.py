# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Partitions the sample space to nodes, ranks, and workers."""

import math
from typing import Optional

import numpy as np


def get_partitions(num_samples: int,
                   num_canonical_nodes: int,
                   num_physical_nodes: int,
                   ranks_per_node: int,
                   workers_per_rank: int,
                   batch_size: Optional[int] = None,
                   drop_first: int = 0):
    """Partition the given number of samples to nodes, ranks, and workers.

    Args:
        num_samples (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.

    Returns:
        NDArray[np.int64]: Partitions (physical nodes x ranks x workers x batches x samples).
    """
    batch_size = batch_size or 1

    # Divide the full dataset sample range into a sample range per canonical node.
    samples_per_canonical_node = (num_samples + num_canonical_nodes - 1) // num_canonical_nodes
    node_ratio = 0
    padding = 0
    if num_canonical_nodes < num_physical_nodes:
        assert not num_physical_nodes % num_canonical_nodes
        node_ratio = num_physical_nodes // num_canonical_nodes
        overflow = samples_per_canonical_node % node_ratio
        if overflow:
            padding = node_ratio - overflow
    padded_samples_per_canonical_node = samples_per_canonical_node + padding

    # Create the initial sample ID matrix.
    #
    # Shape: (canonical nodes, padded samples per canonical node).
    x = np.arange(num_canonical_nodes * padded_samples_per_canonical_node, dtype=np.int64)
    x = x.reshape(num_canonical_nodes, padded_samples_per_canonical_node)

    # Make adustments to replicate the original padding and extending behavior.
    offsets = np.arange(num_canonical_nodes) * padding
    offsets = np.expand_dims(offsets, 1)
    x -= offsets
    starts = np.arange(num_canonical_nodes) * num_samples // num_canonical_nodes
    starts = np.expand_dims(starts, 1)
    x += starts - x[:, :1]
    stops = np.arange(1, 1 + num_canonical_nodes) * num_samples // num_canonical_nodes
    stops = np.expand_dims(stops, 1)
    is_shorts = stops - starts < samples_per_canonical_node
    x[:, samples_per_canonical_node - 1:samples_per_canonical_node] -= is_shorts
    if padding:
        x[:, -padding:] = x[:, -padding - node_ratio + 1 - padding:-padding - node_ratio + 1]

    # Flatten, drop samples that have already been seen, reshape back.
    #
    # Shape: (physical nodes, samples per node).
    x = x.transpose()
    x = x.flatten()
    x = x[drop_first:]
    x = x.reshape(-1, num_physical_nodes)
    x = x.transpose()

    # Interleave the node sample ranges over each node's ranks, padding by repeating the last
    # sample.
    #
    # Shape: (physical nodes, samples per rank, ranks per node).
    overflow = x.shape[1] % ranks_per_node
    if overflow:
        underflow = ranks_per_node - overflow
        last = x[:, -ranks_per_node - underflow + 1:-ranks_per_node + 1]
        x = np.concatenate([x, last], 1)
    x = x.reshape(num_physical_nodes, -1, ranks_per_node)

    # Pad with -1 adequately for reshaping across workers.
    #
    # Shape: (physical nodes, samples per rank, ranks per node).
    overflow = x.shape[1] % workers_per_rank
    rounded_num_samples = math.ceil(
        x.shape[1] / (workers_per_rank * batch_size)) * (workers_per_rank * batch_size)
    overflow = rounded_num_samples - x.shape[1]
    if overflow:
        last = np.full((num_physical_nodes, overflow, ranks_per_node), -1, np.int64)
        x = np.concatenate([x, last], 1)

    # Interleave each rank's padded samples across its workers.
    #
    # Shape: (physical nodes, ranks per node, workers per rank, batches per worker, batch size).
    x = x.transpose(0, 2, 1)
    x = x.reshape(num_physical_nodes, ranks_per_node, -1, workers_per_rank, batch_size)
    return x.transpose(0, 1, 3, 2, 4)
