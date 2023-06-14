import pytest

from streaming.base.partition import get_partitions


def test_partition_walk():
    partition_algo = 'orig'
    num_samples = 1000
    num_canonical_nodes = 176
    num_physical_nodes = 22
    ranks_per_node = 8
    workers_per_rank = 8
    batch_size = 10
    drop_first = 0

    """
    # Requires a lot of RAM and time
    partition_algo = 'orig'
    num_samples = 1999990222
    num_canonical_nodes = 176
    num_physical_nodes = 22
    ranks_per_node = 8
    workers_per_rank = 8
    batch_size = 10
    drop_first = 408112000
    """

    for drop_first in range(0, 500):
        x = get_partitions(partition_algo, num_samples, num_canonical_nodes, num_physical_nodes,
                           ranks_per_node, workers_per_rank, batch_size, drop_first)
