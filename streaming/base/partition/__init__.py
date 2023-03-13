# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples to nodes/ranks/workers for elastically deterministic sample order."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from streaming.base.partition.orig import get_partitions_orig

algos = {
    'orig': get_partitions_orig,
}


def get_partitions(algo: str,
                   num_samples: int,
                   num_canonical_nodes: int,
                   num_physical_nodes: int,
                   ranks_per_node: int,
                   workers_per_rank: int,
                   batch_size: Optional[int] = None,
                   drop_first: int = 0) -> NDArray[np.int64]:
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
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes, ranks per node, workers per rank,
            batches per worker, batch size).
    """
    get = algos[algo]
    return get(num_samples, num_canonical_nodes, num_physical_nodes, ranks_per_node,
               workers_per_rank, batch_size, drop_first)
