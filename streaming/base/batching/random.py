# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples such that batches have samples randomly selected from streams."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from streaming.base.partition import get_partitions
from streaming.base.shuffle import get_shuffle
from streaming.base.world import World

if TYPE_CHECKING:
    from streaming.base.dataset import StreamingDataset

logger = logging.getLogger(__name__)


def generate_work_random_batching(dataset: StreamingDataset, world: World, epoch: int,
                                  sample_in_epoch: int) -> NDArray[np.int64]:
    """Generate this epoch's arrangement of samples for ``random`` batching.

    This is only called in local rank zero. When ``batching_method`` is set to ``per_stream``,
    which is the default case, each batch consists of samples selected at random from across
    all streams.

    Args:
        dataset (StreamingDataset): Dataset to generate the partition for.
        world (World): World state.
        epoch (int): Which epoch it is.
        sample_in_epoch (int): Where we are in the epoch.

    Returns:
        NDArray[np.int64]: The epoch (num physical nodes, ranks per node, workers per rank,
            batches per worker, batch size).
    """
    # Ensure that num_canonical_nodes has been set.
    if dataset.num_canonical_nodes is None:
        raise RuntimeError(f'`num_canonical_nodes` can never be None. ' +
                           f'Provide a positive integer.')

    # Sample each shard of each stream according to their proportions/repeats/samples. This
    # gives us the resampled size of each underlying shard, and a mapping from each fake "big"
    # sample ID to its underlying "small" sample ID.
    shuffle_units, small_per_big = dataset.resample_streams(epoch)

    batch_size = dataset.batch_size
    assert isinstance(batch_size, int), f'Batch size must be an integer. Got {type(batch_size)}.'

    # Partition the global sample space (of resampled "big" sample IDs) into a tensor of shape
    # (num physical nodes, ranks per node, workers per rank, batches per worker, samples per
    # batch) such that we have an elastically deterministic sample order.
    big_ids = get_partitions(dataset.partition_algo, dataset.epoch_size,
                             dataset.num_canonical_nodes, world.num_nodes, world.ranks_per_node,
                             world.workers_per_rank, batch_size, sample_in_epoch,
                             dataset.initial_physical_nodes)

    # If we need to shuffle, shuffle in a node-aware and *underlying* shard-aware way.
    if dataset.shuffle:
        if not isinstance(dataset.shuffle_block_size, int):
            raise TypeError(f'Dataset `shuffle_block_size` must be an integer. ' +
                            f'Got {type(dataset.shuffle_block_size)} instead.')
        shuffle = get_shuffle(dataset.shuffle_algo, shuffle_units, dataset.num_canonical_nodes,
                              dataset.shuffle_seed, epoch, dataset.shuffle_block_size)
        big_ids = np.where(big_ids != -1, shuffle[big_ids], -1)

    # Now that we have partitioning and shuffled with hallucinated "big" sample IDs, we don't
    # need them anymore, and can convert back to underlying "small" sample IDs.
    return np.where(big_ids != -1, small_per_big[big_ids], -1)
