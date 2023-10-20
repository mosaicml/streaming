# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from streaming.base.partition import get_partitions


@pytest.mark.parametrize('partition_algo', ['orig', 'relaxed'])
def test_partition_walk(partition_algo: str):
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
        assert x.shape == (22, 8, 8, 1, 10)


def test_partition_relaxed_resumption():
    # For global batch size 960, which is a highly divisible number, go through all possible
    # values of physical nodes we can train on.
    # Assuming 8 devices per node, we can train on the following numbers of nodes:
    # 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120
    # The initial number of physical nodes is 15, which is also num_canonical_nodes.
    # Without relaxed partitioning, we can only train on the following numbers of nodes with
    # deterministic resumption (with 15 == NCN == initial physical nodes):
    # 1, 3, 5, 15, 30, 60, 120
    # And with orig partitioning, we cannot train on the following numbers of nodes due to the NCN
    # and PN divisibility constraints:
    # 2, 4, 6, 8, 10, 12, 20, 24, 40

    # Make initial partition with with 15 == NCN == initial physical nodes
    initial_physical_nodes = 15
    num_canonical_nodes = 15
    global_batch_size = 960

    num_samples = 10000
    ranks_per_node = 8
    workers_per_rank = 8
    drop_first = 0
    initial_batch_size = global_batch_size // (initial_physical_nodes * ranks_per_node)
    # relaxed partitioning is the same as orig partitioning for the initial partition
    initial_partition = get_partitions('relaxed', num_samples, num_canonical_nodes,
                                       initial_physical_nodes, ranks_per_node, workers_per_rank,
                                       initial_batch_size, drop_first)
    # Get the inorder global batches of the initial partition
    initial_partition = initial_partition.transpose(3, 2, 0, 1, 4).reshape(-1, global_batch_size)
    num_initial_batches = initial_partition.shape[0]

    # For each possible number of physical nodes, get the new partition and check that the inorder
    # global batches are the same with relaxed partitioning.
    resumption_nodes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
    for new_node_num in resumption_nodes:
        new_batch_size = global_batch_size // (new_node_num * ranks_per_node)
        new_partition = get_partitions('relaxed', num_samples, num_canonical_nodes, new_node_num,
                                       ranks_per_node, workers_per_rank, new_batch_size,
                                       drop_first, initial_physical_nodes)
        # Get the inorder global batches of the new partition
        new_partition = new_partition.transpose(3, 2, 0, 1, 4).reshape(-1, global_batch_size)
        for batch_idx in range(num_initial_batches):
            initial_samples = set(initial_partition[batch_idx])
            new_samples = set(new_partition[batch_idx])
            # don't check equality for batches with padding.
            if -1 not in initial_samples and -1 not in new_samples:
                assert initial_samples == new_samples

    # For orig partitioning, test that we can only resume on a limited number of nodes.
    resumption_nodes = [1, 3, 5, 15, 30, 60, 120]
    for new_node_num in resumption_nodes:
        new_batch_size = global_batch_size // (new_node_num * ranks_per_node)
        new_partition = get_partitions('orig', num_samples, num_canonical_nodes, new_node_num,
                                       ranks_per_node, workers_per_rank, new_batch_size,
                                       drop_first)
        # Get the inorder global batches of the new partition
        new_partition = new_partition.transpose(3, 2, 0, 1, 4).reshape(-1, global_batch_size)
        for batch_idx in range(num_initial_batches):
            initial_samples = set(initial_partition[batch_idx])
            new_samples = set(new_partition[batch_idx])
            # don't check equality for batches with padding.
            if -1 not in initial_samples and -1 not in new_samples:
                assert initial_samples == new_samples

    # For orig partitioning, test that we cannot resume on the other node values due to the NCN
    # and PN divisibility constraints.
    resumption_nodes = [2, 4, 6, 8, 10, 12, 20, 24, 40]
    for new_node_num in resumption_nodes:
        new_batch_size = global_batch_size // (new_node_num * ranks_per_node)
        with pytest.raises(ValueError, match=f'Either canonical or physical nodes must be*'):
            _ = get_partitions('orig', num_samples, num_canonical_nodes, new_node_num,
                               ranks_per_node, workers_per_rank, new_batch_size, drop_first)
