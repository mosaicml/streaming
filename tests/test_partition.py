# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import numpy as np
import pytest

from streaming.base.partition import get_partitions
from streaming.base.world import World


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


@pytest.mark.parametrize('num_samples', [405, 812, 1111])
@pytest.mark.parametrize('num_canonical_nodes', [1, 2])
@pytest.mark.parametrize('num_physical_nodes', [2, 8])
@pytest.mark.parametrize('ranks_per_node', [1, 8])
@pytest.mark.parametrize('workers_per_rank', [1, 8])
@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('partition_algo', ['orig', 'relaxed'])
def test_partition_drop_all(
    num_samples: int,
    num_canonical_nodes: int,
    num_physical_nodes: int,
    ranks_per_node: int,
    workers_per_rank: int,
    batch_size: int,
    partition_algo: str,
):
    initial_physical_nodes = None
    if partition_algo == 'relaxed' and num_canonical_nodes == 4 and ranks_per_node == 8:
        num_canonical_nodes = 3
        initial_physical_nodes = 3
        batch_size = batch_size * 3
        num_samples = 3 * num_samples

    # Partitioning should repeat samples so that the epoch size is divisible by the world size.
    # To drop all samples, we need to drop all repeated samples as well.
    world_size = num_physical_nodes * ranks_per_node
    num_repeated_samples = world_size - (num_samples % world_size)
    drop_first = num_samples + num_repeated_samples

    x = get_partitions(partition_algo, num_samples, num_canonical_nodes, num_physical_nodes,
                       ranks_per_node, workers_per_rank, batch_size, drop_first,
                       initial_physical_nodes)
    # Partition should still have the appropriate shape, but without any samples in it.
    assert x.shape == (num_physical_nodes, ranks_per_node, workers_per_rank, 0, batch_size)
    assert x.size == 0


@pytest.mark.parametrize('num_samples', [400, 1000])
@pytest.mark.parametrize('drop_additional', [1, 400])
@pytest.mark.parametrize('num_canonical_nodes', [4])
@pytest.mark.parametrize('num_physical_nodes', [4])
@pytest.mark.parametrize('ranks_per_node', [8])
@pytest.mark.parametrize('workers_per_rank', [8])
@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('partition_algo', ['orig', 'relaxed'])
def test_partition_invalid_drop_first(num_samples: int, drop_additional: int,
                                      num_canonical_nodes: int, num_physical_nodes: int,
                                      ranks_per_node: int, workers_per_rank: int, batch_size: int,
                                      partition_algo: str):

    # Partitioning should repeat samples so that the epoch size is divisible by the world size.
    # For `drop_first` to be invalid, we need to exceed the number of unique samples plus the
    # number of repeated samples.
    world_size = num_physical_nodes * ranks_per_node
    num_repeated_samples = world_size - (num_samples % world_size)
    drop_first = num_samples + num_repeated_samples + drop_additional

    with pytest.raises(ValueError, match=f'Resuming further into the dataset*'):
        _ = get_partitions(partition_algo, num_samples, num_canonical_nodes, num_physical_nodes,
                           ranks_per_node, workers_per_rank, batch_size, drop_first)


@pytest.mark.parametrize('num_samples', [1, 4])
@pytest.mark.parametrize('num_canonical_nodes', [1, 4])
@pytest.mark.parametrize('num_physical_nodes', [1, 4])
@pytest.mark.parametrize('ranks_per_node', [1, 8])
@pytest.mark.parametrize('workers_per_rank', [1, 8])
@pytest.mark.parametrize('batch_size', [4])
def test_partition_small_num_samples(num_samples: int, num_canonical_nodes: int,
                                     num_physical_nodes: int, ranks_per_node: int,
                                     workers_per_rank: int, batch_size: int):
    drop_first = 0
    partition_algo = 'orig'
    x = get_partitions(partition_algo, num_samples, num_canonical_nodes, num_physical_nodes,
                       ranks_per_node, workers_per_rank, batch_size, drop_first)
    assert x.shape == (num_physical_nodes, ranks_per_node, workers_per_rank,
                       math.ceil(num_samples / batch_size), batch_size)


@pytest.mark.parametrize('num_samples', [1, 2, 3])
@pytest.mark.parametrize('num_canonical_nodes', [4])
@pytest.mark.parametrize('num_physical_nodes', [1, 4])
@pytest.mark.parametrize('ranks_per_node', [8])
@pytest.mark.parametrize('workers_per_rank', [8])
@pytest.mark.parametrize('batch_size', [4])
def test_partition_samples_less_than_ncn(num_samples: int, num_canonical_nodes: int,
                                         num_physical_nodes: int, ranks_per_node: int,
                                         workers_per_rank: int, batch_size: int):
    drop_first = 0
    partition_algo = 'orig'

    with pytest.warns(UserWarning, match=f'Trying to partition*'):
        get_partitions(partition_algo, num_samples, num_canonical_nodes, num_physical_nodes,
                       ranks_per_node, workers_per_rank, batch_size, drop_first)


@pytest.mark.parametrize('num_samples', [5, 15, 25, 55, 95, 135])
@pytest.mark.parametrize('num_canonical_nodes', [4])
@pytest.mark.parametrize('num_physical_nodes', [4, 8])
@pytest.mark.parametrize('ranks_per_node', [8])
@pytest.mark.parametrize('workers_per_rank', [8])
@pytest.mark.parametrize('batch_size', [4])
def test_partition_samples_per_node_less_than_ranks_warning(num_samples: int,
                                                            num_canonical_nodes: int,
                                                            num_physical_nodes: int,
                                                            ranks_per_node: int,
                                                            workers_per_rank: int,
                                                            batch_size: int):
    drop_first = 0
    partition_algo = 'orig'

    if num_samples < ranks_per_node * num_physical_nodes:
        with pytest.warns(UserWarning, match=f'Attempting to partition*'):
            get_partitions(partition_algo, num_samples, num_canonical_nodes, num_physical_nodes,
                           ranks_per_node, workers_per_rank, batch_size, drop_first)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            get_partitions(partition_algo, num_samples, num_canonical_nodes, num_physical_nodes,
                           ranks_per_node, workers_per_rank, batch_size, drop_first)


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


@pytest.mark.parametrize('physical_nodes', [1, 4, 7, 13])
@pytest.mark.parametrize('canonical_nodes', [12, 4, 64])
@pytest.mark.parametrize('ranks_per_node', [8])
@pytest.mark.parametrize('workers_per_rank', [1, 8])
@pytest.mark.parametrize('batch_size', [2, 8])
@pytest.mark.parametrize('num_samples', [1024, 16384])
@pytest.mark.parametrize('sample_in_epoch', [0, 256])
@pytest.mark.parametrize('replication', [2, 4, 8])
def test_replication_samples(physical_nodes: int, canonical_nodes: int, ranks_per_node: int,
                             workers_per_rank: int, batch_size: int, num_samples: int,
                             sample_in_epoch: int, replication: int):

    # Make sure canonical nodes works with physical nodes
    if physical_nodes % canonical_nodes != 0 and canonical_nodes % physical_nodes != 0:
        canonical_nodes = physical_nodes

    # Create a World object reflecting the hardware -- the actual nodes and rank
    actual_world = World(physical_nodes, ranks_per_node, workers_per_rank, 0)

    # Create the World object with the given replication factor
    replication_world = actual_world.replicate(replication)

    # Get the sample partition using attributes from the replication World object
    sample_ids = get_partitions('relaxed', num_samples, canonical_nodes,
                                replication_world.num_nodes, replication_world.ranks_per_node,
                                replication_world.workers_per_rank, batch_size, sample_in_epoch,
                                replication_world.num_nodes)

    # Loop over all of the actual nodes/workers/ranks and check that the samples are replicated
    # according to the `replication` factor.
    # Use World objects to correctly index into the sample_ids partition.
    # This is what happens during actual training.
    for n in range(physical_nodes):
        for w in range(workers_per_rank):
            for r in range(0, ranks_per_node, replication):
                # Get the actual and replication World objects for this node/rank/worker.
                # This ensures we have the right rank and worker indices in the World objects.
                baseline_worker_id = (n * ranks_per_node + r) * workers_per_rank + w
                baseline_actual_world = World(physical_nodes, ranks_per_node, workers_per_rank,
                                              baseline_worker_id)
                baseline_replication_world = baseline_actual_world.replicate(replication)
                # Check that the sample ids are all the same for this replication group
                baseline_sample_ids = sample_ids[baseline_replication_world.node,
                                                 baseline_replication_world.rank_of_node,
                                                 baseline_replication_world.worker_of_rank]
                # Loop over the replication group and make sure the sample ids are all the same.
                for i in range(1, replication):
                    repeated_worker_id = (n * ranks_per_node + r + i) * workers_per_rank + w
                    repeated_actual_world = World(physical_nodes, ranks_per_node, workers_per_rank,
                                                  repeated_worker_id)
                    repeated_replication_world = repeated_actual_world.replicate(replication)
                    repeated_sample_ids = sample_ids[repeated_replication_world.node,
                                                     repeated_replication_world.rank_of_node,
                                                     repeated_replication_world.worker_of_rank]
                    assert np.array_equal(baseline_sample_ids, repeated_sample_ids)


@pytest.mark.parametrize('num_nodes', [4])
@pytest.mark.parametrize('ranks_per_node', [8])
@pytest.mark.parametrize('workers_per_rank', [2])
@pytest.mark.parametrize('replication', [-10, 0])
def test_replication_negative(num_nodes: int, ranks_per_node: int, workers_per_rank: int,
                              replication: int):

    orig_world = World(num_nodes, ranks_per_node, workers_per_rank, 0)

    with pytest.raises(ValueError, match=f'Replication factor must be positive.*'):
        orig_world.replicate(replication)


@pytest.mark.parametrize('num_nodes', [4, 8])
@pytest.mark.parametrize('ranks_per_node', [8])
@pytest.mark.parametrize('workers_per_rank', [2])
@pytest.mark.parametrize('replication', [7, 11])
def test_replication_divides_world_size(num_nodes: int, ranks_per_node: int, workers_per_rank: int,
                                        replication: int):

    orig_world = World(num_nodes, ranks_per_node, workers_per_rank, 0)

    with pytest.raises(ValueError, match=f'World size must be divisible by*'):
        orig_world.replicate(replication)
