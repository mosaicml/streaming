# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples to nodes/ranks/workers for elastically deterministic sample order."""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from streaming.base.partition.orig import get_partitions_orig
from streaming.base.partition.relaxed import get_partitions_relaxed

logger = logging.getLogger(__name__)

algos = {
    'orig': get_partitions_orig,
    'relaxed': get_partitions_relaxed,
}


def get_partitions(algo: str,
                   num_samples: int,
                   num_canonical_nodes: int,
                   num_physical_nodes: int,
                   ranks_per_node: int,
                   workers_per_rank: int,
                   batch_size: int,
                   drop_first: int = 0,
                   initial_physical_nodes: Optional[int] = None) -> NDArray[np.int64]:
    """Partition the given number of samples to nodes, ranks, and workers.

    Either canonical or physical nodes must be evenly divisible by the other.

    It is suggested to set num_canonical_nodes higher than your expected number of physical nodes,
    because scaling your number of nodes below that level may result in more shards being used
    across node boundaries due to preserving the same global sample order.

    Args:
        algo (str): Partition algorithm name.
        num_samples (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        batch_size (int): Batch size of DataLoader and dataset, which affects how the dataset is
            partitioned over the workers.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.
        initial_physical_nodes (int, optional): Number of physical nodes at the start of training.
            Defaults to ``None``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes, ranks per node, workers per rank,
            batches per worker, batch size).
    """
    world_size = ranks_per_node * num_physical_nodes
    num_repeated_samples = world_size - (num_samples % world_size)
    if num_samples + num_repeated_samples < drop_first:
        raise ValueError(f'Resuming further into the dataset ({drop_first}) than it has samples ' +
                         f'({num_samples})')

    if num_repeated_samples > 0:
        logger.debug(f'Using {num_repeated_samples} repeated samples to ensure that the epoch ' +
                     f'size is divisible by the number of total devices. This ensures that each ' +
                     f'device contributes the same number of samples per global batch. ')

    get = algos[algo]
    return get(num_samples, num_canonical_nodes, num_physical_nodes, ranks_per_node,
               workers_per_rank, batch_size, drop_first, initial_physical_nodes)
