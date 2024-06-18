# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples to nodes/ranks/workers for elastically deterministic sample order."""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from streaming.base.partition.orig import get_partitions_orig

logger = logging.getLogger(__name__)


def get_partitions_relaxed(num_samples: int,
                           num_canonical_nodes: int,
                           num_physical_nodes: int,
                           ranks_per_node: int,
                           workers_per_rank: int,
                           batch_size: int,
                           drop_first: int = 0,
                           initial_physical_nodes: Optional[int] = None) -> NDArray[np.int64]:
    """Partition the given number of samples to nodes, ranks, and workers.

    Either canonical or physical nodes must be evenly divisible by the other when partitioning over
    the initial number of physical nodes. For partitions during resumption, the only constraint
    is that the global batch size, which remains constant during training, must be evenly divisible
    by the total number of devices, which is num_physical_nodes * ranks_per_node.

    It is suggested to set num_canonical_nodes higher than your expected number of physical nodes,
    because scaling your number of nodes below that level may result in more shards being used
    across node boundaries due to preserving the same global sample order.

    Args:
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
    if initial_physical_nodes is None or (num_physical_nodes <= num_canonical_nodes and
                                          num_canonical_nodes % num_physical_nodes == 0) or \
                                         (num_physical_nodes > num_canonical_nodes and
                                          num_physical_nodes % num_canonical_nodes == 0):
        # Case 1: We are partitioning for the first time. Use the original partitions algorithm,
        # which also requires that NCN be divisible by PN or vice versa.
        # Case 2: PN <= NCN and PN evenly divides NCN. The original partition algo can be used,
        # and will give better downloads per node as well.
        # Case 3: PN > NCN and NCN evenly divides PN. The original partition algo can be used.
        return get_partitions_orig(num_samples, num_canonical_nodes, num_physical_nodes,
                                   ranks_per_node, workers_per_rank, batch_size, drop_first)
    else:
        # First, make a partition over the initial number of physical nodes and device batch size.
        # We assume that ranks_per_node and workers_per_rank stay constant during resumptions.
        global_batch_size = num_physical_nodes * ranks_per_node * batch_size
        initial_total_devices = initial_physical_nodes * ranks_per_node
        # Check for divisibility of the current global batch size and the initial total devices.
        # This should be true since the global batch size should not change in the middle of
        # training.
        if global_batch_size % initial_total_devices != 0:
            raise ValueError(f'A global batch size of {global_batch_size} is not evenly ' +
                             f'divisible by the initial total number of devices of ' +
                             f'{initial_total_devices}. Make sure that when using ' +
                             f'the `relaxed` partitioning algorithm, the global batch size does ' +
                             f'not change during resumption of training.')
        initial_batch_size = global_batch_size // initial_total_devices
        partition = get_partitions_orig(num_samples, num_canonical_nodes, initial_physical_nodes,
                                        ranks_per_node, workers_per_rank, initial_batch_size,
                                        drop_first)

        # Flatten the initial partition in order of traversal.
        # partition was originally (nodes, ranks, workers, batches per worker, batch size)
        # in-order, the dimensions are (batches per worker, workers, nodes, ranks, batch size)
        partition = partition.transpose(3, 2, 0, 1, 4).flatten()

        # Reshape the in-order traversal of the partition to the new physical nodes and batch size.
        partition = partition.reshape(-1, workers_per_rank, num_physical_nodes, ranks_per_node,
                                      batch_size)

        # Re-transpose this partition matrix back to the original format below and return it:
        # (physical nodes, ranks per node, workers per rank, batches per worker, batch size)
        return partition.transpose(2, 3, 1, 0, 4)
