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
                                 drop_first: int = 0,
                                 exact: bool = False) -> NDArray[np.int64]:
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
        exact (bool): If true, epochs are padded exactly, but is sometimes slow. If false, epochs
            contain up to a small amount of extra padding, but is fast. Defaults to ``False``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes x ranks per node x workers per rank
            x batches per worker x batch size per rank).
    """
    if num_canonical_nodes % num_physical_nodes and num_physical_nodes % num_canonical_nodes:
        raise ValueError('One of {canonical nodes, physical nodes} must be evenly divisible by ' +
                         'the other.')

    # Calculate samples per rank and padding.
    num_ranks = num_physical_nodes * ranks_per_node
    min_samples_per_rank = math.floor(dataset_size / num_ranks)
    max_samples_per_rank = math.ceil((dataset_size + dataset_padding) / num_ranks)
    batches_per_worker = math.ceil(max_samples_per_rank / (workers_per_rank * batch_size_per_rank))
    rect_samples_per_rank = workers_per_rank * batches_per_worker * batch_size_per_rank

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

    # Generate the initial sample ID tensor.
    # Sample IDs: (nodes x ranks per node x padded samples per rank).
    starts = node_starts.reshape(-1, 1, 1) + rank_of_node_starts.reshape(1, -1, 1)
    indices = np.arange(rect_samples_per_rank).reshape(1, 1, -1)
    ids = starts + indices * step

    # If we are training on an increased number of nodes, interleave canonical ranks onto
    # physical ranks so that the global order of the samples is preserved.
    if num_physical_nodes < num_canonical_nodes:
        node_ratio = num_canonical_nodes // num_physical_nodes
        ids = ids.reshape(node_ratio, num_physical_nodes, ranks_per_node, -1)
        ids = ids.transpose(1, 3, 2, 0)
        ids = ids.reshape(num_physical_nodes, -1, ranks_per_node)
        ids = ids.transpose(0, 2, 1)
        ids = ids[:, :, :rect_samples_per_rank]
    # Sample IDs: (physical nodes x ranks per node x padded samples per rank).

    # Reassign any sample IDs that extend past the end of the dataset to be within the dataset.
    #
    # We only do this in the narrow band between min_samples_per_rank and max_samples_per_rank
    # because before then doesn't reach the end of the dataset and after then is always dropped.
    #
    # As we are almost certainly on the last shard if we run over, reassign to samples from the
    # last shard in order to avoid a possible extra shard download.
    #
    # We need to keep these samples because each rank must have the same number of samples.
    i = min_samples_per_rank
    j = max_samples_per_rank
    reassign = np.arange(dataset_size - (j - i), dataset_size)
    ids[:, :, i:j] = np.where(ids[:, :, i:j] < dataset_size, ids[:, :, i:j], reassign)

    # Determine where to truncate the samples per rank, if we don't know exactly, via bincounting
    # until all the holes are filled.
    #
    # Any possibility of determining this through some elegant formula is lost by the previous
    # step. The distribution of dataset paddings skews heavily towward zero. The offsets by 1 are
    # to account for the -1 entries.
    if False and min_samples_per_rank < max_samples_per_rank:
        seen = np.bincount(1 + ids[:, :, :min_samples_per_rank].flatten(),
                           minlength=1 + dataset_size)
        for exact_samples_per_rank in range(min_samples_per_rank, max_samples_per_rank + 1):
            if min(seen[1:]):
                break
            if exact_samples_per_rank < max_samples_per_rank:
                seen += np.bincount(1 + ids[:, :, exact_samples_per_rank].flatten(),
                                    minlength=1 + dataset_size)
    else:
        exact_samples_per_rank = max_samples_per_rank

    # Drop all superfluous sample IDs beyond exact_samples_per_rank.
    ids[:, :, exact_samples_per_rank:] = -1

    # If we are mid-epoch, drop the first drop_first samples by flattening into the order that
    # samples would be seen and clipping the samples from the left.
    if drop_first:
        ids = ids.transpose(2, 1, 0)
        ids = ids.flatten()
        ids[:-drop_first] = ids[drop_first:]
        ids[-drop_first:] = -1
        # Return to the original shape.
        ids = ids.reshape(rect_samples_per_rank, ranks_per_node, num_physical_nodes)
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
                                           batch_size_per_rank, False)
        if _are_partitions_valid(dataset_size, ids):
            return dataset_padding
        dataset_padding += 1


def _get_max_dataset_padding(num_canonical_nodes: int, num_physical_nodes: int,
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
                         drop_first: int = 0,
                         exact: bool = False) -> NDArray[np.int64]:
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
        exact (bool): If true, epochs are padded exactly, but is sometimes slow. If false, epochs
            contain up to a small amount of extra padding, but is fast. Defaults to ``False``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes x ranks per node x workers per rank
            x batches per worker x batch size per rank).
    """
    max_dataset_padding = _get_max_dataset_padding(num_canonical_nodes, num_physical_nodes,
                                                   ranks_per_node)
    # exact_dataset_padding = get_dataset_padding_brute(dataset_size, num_canonical_nodes,
    #                                                   num_physical_nodes, ranks_per_node,
    #                                                   workers_per_rank, batch_size_per_rank)
    # assert exact_dataset_padding <= max_dataset_padding
    return _get_partitions_pynum_padded(dataset_size, max_dataset_padding, num_canonical_nodes,
                                        num_physical_nodes, ranks_per_node, workers_per_rank,
                                        batch_size_per_rank, drop_first, exact)
