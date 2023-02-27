# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Partition samples to nodes, ranks, and workers via a pure numpy approach."""

import math

import numpy as np
from numpy.typing import NDArray


def _get_partitions_pynum_padded(dataset_size: int,
                                 dataset_padding: int,
                                 num_canonical_nodes: int,
                                 num_physical_nodes: int,
                                 ranks_per_node: int,
                                 workers_per_rank: int,
                                 batch_size_per_rank: int = 1,
                                 drop_first: int = 0) -> NDArray[np.int64]:
    """Partition the given number of samples to nodes, ranks, and workers.

    Either canonical or physical nodes must be a multiple of the other.

    It is suggested to set num_canonical_nodes higher than your expected number of physical nodes,
    beecause scaling your number of nodes bellow that level may result in shards being used across
    node boundaries in order to preserve the same global sample order.

    Args:
        dataset_size (int): Dataset size.
        dataset_padding (int): How much to pad the dataset size when calculating samples per rank.
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
    unpadded_samples_per_rank = math.floor(dataset_size / num_ranks)
    samples_per_rank = math.ceil((dataset_size + dataset_padding) / num_ranks)
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

    # Drop all unwanted sample IDs hallucinated past the end of each rank's partition.
    ids[:, :, samples_per_rank:] = -1

    # Of those sample IDs that remain (which need to be present to keep samples balanced across
    # ranks), reassign the ones that extend past the end of the dataset to the last sample (as we
    # were probably on the last shard, in order to avoid an extra shard download).
    start = unpadded_samples_per_rank
    stop = samples_per_rank
    ids[:, :, start:stop] = np.where(ids[:, :, start:stop] < dataset_size, ids[:, :, start:stop],
                                     dataset_size - 1)

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


def _are_partitions_valid(dataset_size: int, ids: NDArray[np.int64]) -> bool:
    """Check whether the generated partitioning of sample IDs is valid (contains every ID).

    Args:
        dataset_size (int): Dataset size.
        ids (NDArray[np.int64]): Sample ID partitioning result.

    Returns:
        bool: Whether the partitioning is valid.
    """
    seen = np.zeros(dataset_size, np.uint8)
    seen[ids] = 1
    return bool(min(seen))


def get_dataset_padding_brute(dataset_size: int,
                              num_canonical_nodes: int,
                              num_physical_nodes: int,
                              ranks_per_node: int,
                              workers_per_rank: int,
                              batch_size_per_rank: int = 1) -> int:
    """Determine the dataset padding empiricially.

    This method is not recommended for datasets of any size.

    Args:
        dataset_size (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of data loader worker per rank.
        batch_size_per_rank (int): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``1``.

    Returns:
        int: Dataset padding.
    """
    dataset_padding = 0
    while True:
        ids = _get_partitions_pynum_padded(dataset_size, dataset_padding, num_canonical_nodes,
                                           num_physical_nodes, ranks_per_node, workers_per_rank,
                                           batch_size_per_rank)
        if _are_partitions_valid(dataset_size, ids):
            return dataset_padding
        dataset_padding += 1


def _get_dataset_padding_approx(num_canonical_nodes: int, num_physical_nodes: int,
                                ranks_per_node: int) -> int:
    """Approximate the dataset padding analytically.

    The returned value may be too high, by at most ``num_canonical_nodes - num_physical_nodes``. It
    will never be too low.

    This method was derived empirically to replicate the results of many millions of partitions.

    Args:
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.

    Returns:
        int: Dataset padding.
    """
    if num_canonical_nodes <= num_physical_nodes:
        return 0

    if not ranks_per_node % (num_canonical_nodes // num_physical_nodes):
        return 0

    return num_canonical_nodes - num_physical_nodes


def get_partitions_pynum(dataset_size: int,
                         num_canonical_nodes: int,
                         num_physical_nodes: int,
                         ranks_per_node: int,
                         workers_per_rank: int,
                         batch_size_per_rank: int = 1,
                         drop_first: int = 0) -> NDArray[np.int64]:
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
    dataset_padding = _get_dataset_padding_approx(num_canonical_nodes, num_physical_nodes,
                                                  ranks_per_node)
    # exact_dataset_padding = get_dataset_padding_brute(dataset_size, num_canonical_nodes,
    #                                                   num_physical_nodes, ranks_per_node,
    #                                                   workers_per_rank, batch_size_per_rank)
    # assert exact_dataset_padding <= dataset_padding
    return _get_partitions_pynum_padded(dataset_size, dataset_padding, num_canonical_nodes,
                                        num_physical_nodes, ranks_per_node, workers_per_rank,
                                        batch_size_per_rank, drop_first)
