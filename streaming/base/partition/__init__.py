# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Partition samples to nodes, ranks, and workers."""

from streaming.base.partition.pynum import get_partitions_pynum
from streaming.base.partition.pypy import get_partitions_pypy

algos = {
    'pypy': get_partitions_pypy,
    'pynum': get_partitions_pynum,
}


def get_partitions(algo: str,
                   dataset_size: int,
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
        algo (str): Partitioning algortihm name.
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
    get = algos[algo]
    return get(dataset_size, num_canonical_nodes, num_physical_nodes, ranks_per_node,
               workers_per_rank, batch_size_per_rank, drop_first)
