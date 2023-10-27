# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
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

    # For the limited number of physical nodes that satisfy the NCN and PN divisibility constraints
    # of `orig` partitioning, make sure that resumption with `orig` partitioning has the same
    # sample order as the initial partition, and also that resumption with `relaxed` partitioning
    # outputs the same sample order as `orig`.
    orig_resumption_nodes = [1, 3, 5, 15, 30, 60, 120]

    for new_node_num in orig_resumption_nodes:
        new_batch_size = global_batch_size // (new_node_num * ranks_per_node)
        new_partition_orig = get_partitions('orig', num_samples, num_canonical_nodes, new_node_num,
                                            ranks_per_node, workers_per_rank, new_batch_size,
                                            drop_first, initial_physical_nodes)
        new_partition_relaxed = get_partitions('relaxed', num_samples, num_canonical_nodes,
                                               new_node_num, ranks_per_node, workers_per_rank,
                                               new_batch_size, drop_first, initial_physical_nodes)
        # Get the inorder global batches of the new partition with both partitioning algos.
        new_partition_orig = new_partition_orig.transpose(3, 2, 0, 1, 4)
        new_partition_orig = new_partition_orig.reshape(-1, global_batch_size)
        new_partition_relaxed = new_partition_relaxed.transpose(3, 2, 0, 1, 4)
        new_partition_relaxed = new_partition_relaxed.reshape(-1, global_batch_size)

        # Check equality between the global batches of the new partition with `orig` partitioning
        # and the initial partition's global batches. While the global batches may not have samples
        # in the same order, they should still contain the same samples overall.
        for batch_idx in range(num_initial_batches):
            initial_samples = set(initial_partition[batch_idx])
            new_samples = set(new_partition_orig[batch_idx])
            # don't check equality for batches with padding.
            if -1 not in initial_samples and -1 not in new_samples:
                assert initial_samples == new_samples

        # Check equality between the partition with `orig` partitioning and the partition with
        # `relaxed` partitioning. The whole arrays should be equal since `relaxed` partitioning
        # should default to `orig` partitioning when the NCN and PN divisibility constraints are
        # fulfilled.
        assert np.array_equal(new_partition_orig, new_partition_relaxed)

    # For orig partitioning, test that we cannot resume on the node values below due to the NCN
    # and PN divisibility constraints. Make sure that we can resume with `relaxed` partitioning,
    # and that the resulting sample partition, when modified to be global batches in order of
    # traversal, is the exact same as the original sample partition's global batches.
    relaxed_resumption_nodes = [2, 4, 6, 8, 10, 12, 20, 24, 40]
    for new_node_num in relaxed_resumption_nodes:
        new_batch_size = global_batch_size // (new_node_num * ranks_per_node)

        # Test that `orig` partition fails with this number of physical_nodes.
        with pytest.raises(ValueError, match=f'Either canonical or physical nodes must be*'):
            _ = get_partitions('orig', num_samples, num_canonical_nodes, new_node_num,
                               ranks_per_node, workers_per_rank, new_batch_size, drop_first)

        # Test that `relaxed` partition succeeds with this number of physical_nodes.
        new_partition_relaxed = get_partitions('relaxed', num_samples, num_canonical_nodes,
                                               new_node_num, ranks_per_node, workers_per_rank,
                                               new_batch_size, drop_first, initial_physical_nodes)
        # Get the inorder global batches of the new partition
        new_partition_relaxed = new_partition_relaxed.transpose(3, 2, 0, 1, 4)
        new_partition_relaxed = new_partition_relaxed.reshape(-1, global_batch_size)
        # The new partition's global batches with relaxed partitioning should be the exact same
        # as the initial partitioning's global batches.
        assert np.array_equal(new_partition_relaxed, initial_partition)


@pytest.mark.parametrize('physical_nodes', [1, 4])
@pytest.mark.parametrize('canonical_nodes', [4])
@pytest.mark.parametrize('ranks_per_node', [8])
@pytest.mark.parametrize('workers_per_rank', [1, 8])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('num_samples', [512, 1024])
@pytest.mark.parametrize('partition_algo', ['orig', 'relaxed'])
def test_partition_nodrop_norepeat(physical_nodes: int, canonical_nodes: int, ranks_per_node: int,
                                   workers_per_rank: int, batch_size: int, num_samples: int,
                                   partition_algo: str):
    # This test uses values `num_samples` that are divisible by canonical_nodes, physical_nodes,
    # ranks_per_node, workers_per_rank, and batch_size, so that no samples are dropped or repeated.
    drop_first = 0

    partition = get_partitions(partition_algo, num_samples, canonical_nodes, physical_nodes,
                               ranks_per_node, workers_per_rank, batch_size, drop_first)

    # Get the inorder global batches of the partition
    global_batch_size = batch_size * ranks_per_node * physical_nodes
    partition = partition.transpose(3, 2, 0, 1, 4).reshape(-1, global_batch_size)

    # Append samples seen from the partition to this list. If the length of this list is equal to
    # the number of original samples, and the set of samples seen equals the set of numbers from
    # [0, num_samples), we know that no samples were dropped or repeated.
    samples_seen = []
    samples_seen_set = set()
    for global_batch in partition:
        for sample in global_batch:
            samples_seen.append(sample)
            samples_seen_set.add(sample)

    assert len(samples_seen) == num_samples
    assert samples_seen_set == set(range(num_samples))
