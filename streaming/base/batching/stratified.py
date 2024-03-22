# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples so each batch has the same amount of samples from each stream."""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from streaming.base.partition import get_partitions
from streaming.base.shuffle import get_shuffle
from streaming.base.world import World

if TYPE_CHECKING:
    from streaming.base.dataset import StreamingDataset

logger = logging.getLogger(__name__)


def generate_work_stratified_batching(dataset: StreamingDataset, world: World, epoch: int,
                                      sample_in_epoch: int) -> NDArray[np.int64]:
    """Generate the epoch's sample arrangement for ``stratified`` batching method.

    This is only called in local rank zero. When ``batching_method`` is set to ``stratified``,
    every single batch is divided between streams in the same proportions.

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

    # First, for each stream, sample each shard of the stream using proportions/repeats/samples.
    # We obtain the resampled size of each shard in the stream and a mapping from the
    # training "big" sample ID to the underlying shard "small" sample ID.
    # Then, we also partition each stream's samples over nodes/devices/workers.
    # We handle sample_in_epoch (for resumption) at the end.

    batch_size = dataset.batch_size
    assert isinstance(batch_size, int), f'Batch size must be an integer. Got {type(batch_size)}.'

    global_batch_size = batch_size * world.ranks_per_node * world.num_nodes
    partition_per_stream = []
    batch_portion_per_stream = []
    stream_proportions = []
    for stream_id, stream in enumerate(dataset.streams):
        # find how many samples in each global batch are from each stream.
        batch_portion = int(stream.proportion * global_batch_size)
        stream_proportions.append(stream.proportion)
        batch_portion_per_stream.append(batch_portion)

        shuffle_units, small_per_big = dataset.resample_streams(epoch, stream_id)
        samples_in_stream = len(small_per_big)
        # The partition for each stream is constructed with batch size 1 and 1 physical node
        # in order to make sure that the sample order from each batch stays the same
        # We later reshape these partitions to the correct batch size per stream.
        # We also handle used samples (drop_first) at the end.
        stream_partition = get_partitions(dataset.partition_algo, samples_in_stream,
                                          dataset.num_canonical_nodes, 1, world.ranks_per_node,
                                          world.workers_per_rank, 1, 0,
                                          dataset.initial_physical_nodes)
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
        # The small_per_big array corresponds to indices of samples per shard of each stream.
        # So each sample ID in the stream's partition already corresponds to the sample ID
        # in the right shard.
        partition_per_stream.append(
            np.where(stream_partition != -1, small_per_big[stream_partition], -1))

    # The sum of batch portion sizes per stream might not equal the global batch size.
    batch_portion_per_stream = np.array(batch_portion_per_stream)
    batch_parts_sum = np.sum(batch_portion_per_stream)
    if batch_parts_sum != global_batch_size:
        missing_samples = global_batch_size - batch_parts_sum
        # Select the streams that should get the extra samples by seeing which streams were
        # "closest" to having an additional sample, but did not because int conversion rounds down.
        leftover_batch_part_sizes = global_batch_size * np.array(
            stream_proportions) - batch_portion_per_stream
        # We have to flip the array since argsort is in ascending order, and we want to prioritize
        # streams that were closest to getting a sample (highest leftover batch part size.)
        # Then, only get the top missing_samples number of streams to increment batch part sizes.
        stream_size_increment_ids = np.flip(
            np.argsort(leftover_batch_part_sizes))[:missing_samples]
        batch_portion_per_stream[stream_size_increment_ids] += 1

    # Check if any stream's batch portion is 0. If so, raise ValueError.
    for stream_id, batch_portion in enumerate(batch_portion_per_stream):
        if batch_portion <= 0:
            raise ValueError(
                f'Number of samples for stream {stream_id} is {batch_portion} because the portion '
                +
                f'of this stream in the global batch, which is of size {global_batch_size}, is ' +
                f'too low. Please increase the global batch size or increase the porportion of ' +
                f'total samples that come from stream {stream_id}.')

    # We now merge the partitions from each stream to get our final partition over all
    # streams, where every single global batch has the same sample composition from the streams.
    # The total number of batches we can make is constrained by the min batch parts available
    # from any one stream.
    min_batch_parts = np.inf
    batches_from_partitions = []
    for i, partition in enumerate(partition_per_stream):
        # Reshape the partition to batch portion per stream in order of traversal, and count
        # only batches without -1 in them. Before reshaping, make sure number of samples in each
        # stream is divisible by the batch_portion_per_stream.
        batch_parts_inorder = partition.transpose(3, 2, 0, 1, 4).flatten()
        samples_in_stream_partition = batch_parts_inorder.size
        if samples_in_stream_partition % batch_portion_per_stream[i] != 0:
            padding_samples = batch_portion_per_stream[i] - (samples_in_stream_partition %
                                                             batch_portion_per_stream[i])
            batch_parts_inorder = np.concatenate(
                (batch_parts_inorder, np.full(padding_samples, -1)))
        # Reshape to get batch portions from this stream, in order of traversal.
        batch_parts_inorder = batch_parts_inorder.reshape(-1, batch_portion_per_stream[i])
        num_full_batches = np.count_nonzero(np.min(batch_parts_inorder, axis=1) >= 0)
        if num_full_batches != batch_parts_inorder.shape[0]:
            logger.warning(
                'Because of the `stratified` batching method, some batches with an inadequate \
                number of samples from stream with index ' + str(i) + ' are being dropped.')
        if num_full_batches < min_batch_parts:
            min_batch_parts = num_full_batches
        batches_from_partitions.append(batch_parts_inorder)

    # clip the partitions from all streams to only have min_batch_parts batch parts.
    batches_from_partitions = [
        batch_partition[:min_batch_parts] for batch_partition in batches_from_partitions
    ]

    # Concatenate the batch parts from every stream to form all the global batches
    all_partition_batches = np.concatenate(batches_from_partitions, axis=1)

    # If applicable we resume right after the most recently used full global batch.
    if sample_in_epoch % global_batch_size != 0:
        warnings.warn('Because of the `stratified` batching method, resumption may \
            only occur on a sample that is a multiple of the current global batch \
            size of ' + str(global_batch_size) + '. Resuming training after the most \
            recently finished global batch. Set ' + str(global_batch_size) + ' to \
            the original value for deterministic resumption.')

    # Discard previous batches that may have already finished
    resumption_batch = sample_in_epoch // global_batch_size
    all_partition_batches = all_partition_batches[resumption_batch:]

    # Add padding batches if needed to ensure that we have an even number of
    # batches per worker/rank/node
    current_samples = all_partition_batches.size
    divisibility_requirement = world.num_nodes * world.ranks_per_node * \
        world.workers_per_rank * batch_size
    if current_samples % divisibility_requirement != 0:
        samples_needed = divisibility_requirement - (current_samples % divisibility_requirement)
        padding_batches_needed = samples_needed // global_batch_size
        all_partition_batches = np.concatenate(
            (all_partition_batches, np.full((padding_batches_needed, global_batch_size), -1)))

    # Reverse the transposition and reshape from earlier.
    # Final result is (physical nodes, ranks, workers, batches, batch size).
    return all_partition_batches.reshape(-1, world.workers_per_rank, world.num_nodes,
                                         world.ranks_per_node,
                                         batch_size).transpose(2, 3, 1, 0, 4)
