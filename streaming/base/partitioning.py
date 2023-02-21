# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Partitions the sample space to nodes, ranks, and workers."""

import math

import numpy as np


def get_partitions_slow(dataset_size: int,
                        num_canonical_nodes: int,
                        num_physical_nodes: int,
                        ranks_per_node: int,
                        workers_per_rank: int,
                        batch_size_per_rank: int = 1,
                        drop_first: int = 0):
    """Partition the given number of samples to nodes, ranks, and workers.

    Either canonical or physical nodes must be a multiple of the other.

    It is suggested to set num_canonical_nodes higher than your expected number of physical nodes,
    beecause scaling your number of nodes bellow that level may result in shards being used across
    node boundaries in order to preserve the same global sample order.

    Args:
        dataset_size (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        batch_size_per_rank (int): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``1``.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes x ranks per node x workers per rank
            x batches per worker x batch size per rank).
    """
    # Divide the full dataset sample range into sample range per canonical node.
    arrs = []
    for i in range(num_canonical_nodes):
        start = i * dataset_size // num_canonical_nodes
        stop = (i + 1) * dataset_size // num_canonical_nodes
        arr = np.arange(start, stop)
        arrs.append(arr)

    # Make the spans equal length by repeating the last sample if too short.
    # -> Shape: (canonical nodes x samples).
    max_len = max(map(len, arrs))
    for i, x in enumerate(arrs):
        if len(x) < max_len:
            last = np.array([x[-1]])
            arrs[i] = np.concatenate([x, last])
    x = np.stack(arrs)

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

    # Drop samples that have already been seen and reshape.
    # -> Shape: (physical nodes x samples).
    x = x.transpose()
    x = x.flatten()
    x = x[drop_first:]
    x = x.reshape(-1, num_physical_nodes)
    x = x.transpose()

    # Interleave the node sample ranges over each node's rank, padding by repeating last sample.
    # -> Shape: (physical nodes x samples x ranks).
    too_many = x.shape[1] % ranks_per_node
    if too_many:
        too_few = ranks_per_node - too_many
        last = x[:, -ranks_per_node - too_few + 1:-ranks_per_node + 1]
        x = np.concatenate([x, last], 1)
    x = x.reshape(num_physical_nodes, -1, ranks_per_node)

    # Interleave each rank's samples across its workers, padding with -1.
    # -> Shape: (physical nodes x samples x workers x ranks).
    too_many = x.shape[1] % workers_per_rank
    # Make the number of samples multiple of batch size and workers per rank
    rounded_num_samples = math.ceil(x.shape[1] / (workers_per_rank * batch_size_per_rank)) * \
        (workers_per_rank * batch_size_per_rank)
    too_many = rounded_num_samples - x.shape[1]
    if too_many:
        last = np.full((num_physical_nodes, too_many, ranks_per_node), -1, np.int64)
        x = np.concatenate([x, last], 1)

    # -> Shape: (physical nodes x ranks per node x samples)
    x = x.transpose(0, 2, 1)
    x = x.reshape(num_physical_nodes, ranks_per_node, -1, batch_size_per_rank)
    x = np.concatenate(
        [x[:, :, np.arange(i, x.shape[2], workers_per_rank), :] for i in range(workers_per_rank)],
        axis=2)

    # -> Shape: (physical nodes x ranks per node x workers per rank x batches per worker x batch size
    #            per rank).
    return x.reshape(num_physical_nodes, ranks_per_node, workers_per_rank, -1, batch_size_per_rank)


def get_partitions_fast(dataset_size: int,
                        num_canonical_nodes: int,
                        num_physical_nodes: int,
                        ranks_per_node: int,
                        workers_per_rank: int,
                        batch_size_per_rank: int = 1,
                        drop_first: int = 0):
    """Partition the given number of samples to nodes, ranks, and workers.

    Either canonical or physical nodes must be a multiple of the other.

    It is suggested to set num_canonical_nodes higher than your expected number of physical nodes,
    beecause scaling your number of nodes bellow that level may result in shards being used across
    node boundaries in order to preserve the same global sample order.

    Args:
        dataset_size (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of data loader worker per rank.
        batch_size_per_rank (int): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``1``.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes x ranks per node x workers per rank
            x batches per worker x batch size per rank).
    """
    if num_canonical_nodes % num_physical_nodes and num_physical_nodes % num_canonical_nodes:
        raise ValueError('One of {canonical nodes, physical nodes} must be evenly divisible by ' +
                         'the other.')

    # Calculate samples per rank and padding.
    num_ranks = num_physical_nodes * ranks_per_node
    samples_per_rank = math.ceil(dataset_size / num_ranks)
    batches_per_worker = math.ceil(samples_per_rank / (workers_per_rank * batch_size_per_rank))
    padded_samples_per_rank = workers_per_rank * batches_per_worker * batch_size_per_rank

    # Calculate starts and steps in terms of canonical nodes.
    node_starts = dataset_size * np.arange(num_canonical_nodes) // num_canonical_nodes
    rank_of_node_starts = np.arange(ranks_per_node)
    step = ranks_per_node

    # If we are training on a reduced number of nodes, scale starts and steps accordingly.
    if num_canonical_nodes < num_physical_nodes:
        node_ratio = num_physical_nodes // num_canonical_nodes
        node_starts = np.tile(node_starts, node_ratio)
        node_starts += np.arange(node_ratio).repeat(num_canonical_nodes)
        rank_of_node_starts *= node_ratio
        step *= node_ratio

    # Generate the initial sample IDs tensor.
    # Sample IDs: (nodes x ranks per node x padded samples per rank).
    starts = node_starts.reshape(-1, 1, 1) + rank_of_node_starts.reshape(1, -1, 1)
    indices = np.arange(padded_samples_per_rank).reshape(1, 1, -1)
    ids = starts + indices * step

    # If we are training on an increased number of nodes, interleave canonical ranks onto
    # physical ranks so that the global order of the samples is preserved.
    if num_physical_nodes < num_canonical_nodes:
        node_ratio = num_canonical_nodes // num_physical_nodes
        ids = ids.reshape(node_ratio, num_physical_nodes, ranks_per_node, -1)
        ids = ids.transpose(1, 3, 2, 0)
        ids = ids.reshape(num_physical_nodes, -1, ranks_per_node)
        ids = ids.transpose(0, 2, 1)
        ids = ids[:, :, :padded_samples_per_rank]
    # Sample IDs: (physical nodes x ranks per node x padded samples per rank).

    # Reassign sample IDs that need to be present to keep samples balanced across ranks, but would
    # extend past the end of the dataset.
    second_to_last = ids[:, :, samples_per_rank - 2]
    last = ids[:, :, samples_per_rank - 1]
    ids[:, :, samples_per_rank - 1] = np.where(last < dataset_size, last, second_to_last)
    # Drop all unwanted sample IDs hallucinated past the end of each rank's partition.
    ids[:, :, samples_per_rank:] = -1

    # If we are mid-epoch, drop the first drop_first samples by flattening into the order that
    # samples would be seen and clipping the samples from the left.
    if drop_first:
        ids = ids.transpose(2, 1, 0)
        ids = ids.flatten()
        ids[:-drop_first] = ids[drop_first:]
        ids[-drop_first:] = -1
        # Return to the original shape.
        ids = ids.reshape(padded_samples_per_rank, ranks_per_node, num_physical_nodes)
        ids = ids.transpose(2, 1, 0)

    # Partition samples per rank across each rank's workers and workers' batches.
    ids = ids.reshape(num_physical_nodes, ranks_per_node, batches_per_worker, workers_per_rank,
                      batch_size_per_rank)
    return ids.transpose(0, 1, 3, 2, 4)
    # Sample IDs: (physical nodes x ranks per node x workers per rank x batches per worker x batch size
    # per rank).


get_partitions = get_partitions_fast
