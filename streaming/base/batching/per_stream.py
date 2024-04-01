# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples such that batches have samples only from a single stream."""
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


def generate_work_per_stream_batching(dataset: StreamingDataset, world: World, epoch: int,
                                      sample_in_epoch: int) -> NDArray[np.int64]:
    """Generate this epoch's arrangement of samples for ``per_stream`` batching.

    This is only called in local rank zero. When ``batching_method`` is set to ``per_stream``,
    each batch consists of samples from only one stream.

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

    # First, for each stream, sample each shard of the stream according to
    # stream proportions/repeats/samples. We obtain the resampled size of each shard
    # in the stream and a mapping from the training "big" sample ID to the underlying
    # shard "small" sample ID. Then, we also partition each stream's samples over
    # nodes/devices/workers. We handle sample_in_epoch (for resumption) at the end.
    partition_per_stream = []

    batch_size = dataset.batch_size
    assert isinstance(batch_size, int), f'Batch size must be an integer. Got {type(batch_size)}.'

    for stream_id, stream in enumerate(dataset.streams):
        shuffle_units, small_per_big = dataset.resample_streams(epoch, stream_id)
        samples_in_stream = len(small_per_big)
        stream_partition = get_partitions(dataset.partition_algo, samples_in_stream,
                                          dataset.num_canonical_nodes, world.num_nodes,
                                          world.ranks_per_node, world.workers_per_rank, batch_size,
                                          0, dataset.initial_physical_nodes)
        if dataset.shuffle:
            # Ratio of stream's shuffle block size to overall shuffle block size should be the
            # same as the ratio of the stream's samples to overall samples.
            # This ensures that the overall training shuffle block size is still approximately
            # equal to what is set by the user, and allows for reasoning about cache_limit as well.
            if not isinstance(dataset.shuffle_block_size, int):
                raise TypeError(f'Dataset `shuffle_block_size` must be an integer. ' +
                                f'Got {type(dataset.shuffle_block_size)} instead.')
            shuffle_block_portion = int(dataset.shuffle_block_size * stream.proportion)
            stream_shuffle = get_shuffle(dataset.shuffle_algo, shuffle_units,
                                         dataset.num_canonical_nodes, dataset.shuffle_seed, epoch,
                                         shuffle_block_portion)
            stream_partition = np.where(stream_partition != -1, stream_shuffle[stream_partition],
                                        -1)
        # The small_per_big array already corresponds to indices of samples per shard of each
        # stream. So each sample ID in the stream's partition already corresponds to the sample ID
        # in the right shard.
        partition_per_stream.append(
            np.where(stream_partition != -1, small_per_big[stream_partition], -1))

    # We now merge the partitions from each stream to get our final partition over all
    # streams, where each global batch has samples only from a single stream.
    # Partitions are (physical nodes, ranks, workers, batches per worker, batch size).
    batches_per_stream = []
    batches_from_partitions = []
    for stream_idx, partition in enumerate(partition_per_stream):
        # Reshape the partition to be global batches in order of traversal.
        # We only count only batches without -1 in them.
        global_batches_inorder = partition.transpose(3, 2, 0, 1, 4).reshape(
            -1, batch_size * world.ranks_per_node * world.num_nodes)
        num_full_batches = np.count_nonzero(np.min(global_batches_inorder, axis=1) >= 0)
        batches_per_stream.append(num_full_batches)
        if num_full_batches != global_batches_inorder.shape[0]:
            logger.warning(
                f'Because of the `per_stream` batching method, some batches with an inadequate ' +
                f'number of samples from stream with index {stream_idx} will be dropped.')
        if num_full_batches > 0:
            batches_from_partitions.append(global_batches_inorder[:num_full_batches])
        else:
            logger.warning(f'Stream with index {stream_idx} does not have an adequate number of ' +
                           f'samples to construct a complete global batch. Training will occur ' +
                           f'without any samples from this stream!')

    # Combine all global batches from all streams into one array.
    all_partition_batches = np.concatenate(batches_from_partitions)

    # Shuffle seed changes with every epoch, so the order of streams in our batches also changes.
    epoch_rng = np.random.default_rng(dataset.shuffle_seed + epoch)

    # stream_origins is an array that tells us which stream each batch is using.
    stream_origins = np.concatenate(
        [np.full(n_batch, i) for i, n_batch in enumerate(batches_per_stream)])
    epoch_rng.shuffle(stream_origins)

    # Now, we want the batch_indices array to correctly index into the all_partition_batches
    # array according to stream_origins in order to get our final batch order.
    # For each stream, we want to traverse its batches in the same order as given in its partition.
    batch_indices = np.zeros(stream_origins.shape[0]).astype(np.int64)
    batch_offset = 0
    for i, n_batch in enumerate(batches_per_stream):
        # Update batch_indices for the one stream at a time.
        batch_indices[stream_origins == i] += batch_offset + np.arange(n_batch)
        batch_offset += n_batch

    # Rearrange all_partition_batches by the batch_indices we have obtained.
    all_partition_batches = all_partition_batches[batch_indices]

    # If applicable we resume right after the most recently used full global batch.
    global_batch_size = batch_size * world.num_nodes * world.ranks_per_node
    if sample_in_epoch % global_batch_size != 0:
        logger.warning(
            'Because of the `per_stream` batching method, resumption may only occur on a sample \
            that is a multiple of the current global batch size of ' + str(global_batch_size) +
            '. Resuming training after the most recently finished global batch.')

    # Discard previous batches that may have already finished
    resumption_batch = sample_in_epoch // global_batch_size
    all_partition_batches = all_partition_batches[resumption_batch:]

    # Add padding batches if needed to ensure that we have an even number of
    # batches per worker/rank/node.
    current_samples = all_partition_batches.size
    divisibility_requirement = world.num_nodes * world.ranks_per_node * \
          world.workers_per_rank * batch_size
    if current_samples % divisibility_requirement != 0:
        samples_needed = divisibility_requirement - (current_samples % divisibility_requirement)
        padding_batches_needed = samples_needed // global_batch_size
        all_partition_batches = np.concatenate(
            (all_partition_batches, np.full((padding_batches_needed, global_batch_size), -1)))

    # Reverse the transposition and reshape from earlier.
    # Final result is (physical nodes, ranks, worker, batches per worker, batch size).
    return all_partition_batches.reshape(-1, world.workers_per_rank, world.num_nodes,
                                         world.ranks_per_node,
                                         batch_size).transpose(2, 3, 1, 0, 4)
