# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples such that device batches have samples only from a single stream."""
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


def generate_work_device_per_stream_batching(dataset: StreamingDataset, world: World, epoch: int,
                                             sample_in_epoch: int) -> NDArray[np.int64]:
    """Generate this epoch's arrangement of samples for ``device_per_stream`` batching.

    This is only called in local rank zero. When ``batching_method`` is set to ``device_per_stream``,
    each device batch consists of samples from only one stream.

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

    if dataset.num_canonical_nodes % world.num_nodes != 0:
        raise ValueError(
            f'For `device_per_stream` batching, num_canonical_nodes must be divisible by physical nodes. '
            +
            f'Got {dataset.num_canonical_nodes} canonical nodes and {world.num_nodes} physical nodes.'
        )

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
        # The partition for each stream is constructed with batch size 1 and physical nodes
        # equal to canonical nodes in order to make sure that the sample order from each batch
        # stays the same even when the number of physical nodes & batch size change.
        # We later reshape these partitions to the correct batch size per stream.
        # We also handle used samples (drop_first) at the end.
        stream_partition = get_partitions(dataset.partition_algo, samples_in_stream,
                                          dataset.num_canonical_nodes, dataset.num_canonical_nodes,
                                          world.ranks_per_node, world.workers_per_rank, 1, 0,
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
        # The small_per_big array already corresponds to indices of samples per shard of each
        # stream. So each sample ID in the stream's partition already corresponds to the sample ID
        # in the right shard.
        partition_per_stream.append(
            np.where(stream_partition != -1, small_per_big[stream_partition], -1))

    # We now merge the partitions from each stream to get a partition over all streams
    # but for each physical node, where each device batch has samples only from a single stream.
    # Original partitions are (physical nodes, ranks, workers, batches per worker, batch size).
    # Since we have partitioned each stream with physical nodes equal to canonical nodes,
    # we index into the partition to get each stream's portion of samples for each physical node.
    batches_per_stream = []
    batches_from_partitions = []
    ncn_per_node = dataset.num_canonical_nodes // world.num_nodes
    for node in range(world.num_nodes):
        # We keep track of per-node, per-stream device batches, and the number of batches per-node, per-stream.
        per_node_stream_partitions = []
        per_node_batches_per_stream = []
        for stream_idx, partition in enumerate(partition_per_stream):
            # Reshape the partition to be device batches in order of traversal.
            # We only count only batches without -1 in them.
            stream_samples_inorder = partition[node * ncn_per_node:(node + 1) *
                                               ncn_per_node].transpose(3, 2, 0, 1, 4).flatten()
            # Pad samples to make sure they are divisible by the device batch size.
            padding_samples = batch_size - (stream_samples_inorder.size % batch_size)
            stream_samples_inorder = np.concatenate(
                (stream_samples_inorder, np.full(padding_samples, -1)))
            # Reshape samples to be device batches in order of traversal.
            stream_samples_inorder = stream_samples_inorder.reshape(-1, batch_size)
            num_full_batches = np.count_nonzero(np.min(stream_samples_inorder, axis=1) >= 0)
            per_node_batches_per_stream.append(num_full_batches)
            if num_full_batches != stream_samples_inorder.shape[0]:
                logger.warning(
                    f'Because of the `device_per_stream` batching method, some batches with an inadequate '
                    + f'number of samples from stream with index {stream_idx} will be dropped.')
            if num_full_batches > 0:
                per_node_stream_partitions.append(stream_samples_inorder[:num_full_batches])
            else:
                raise ValueError(
                    f'Stream with index {stream_idx} does not have an adequate number of ' +
                    f'samples to construct even a single device batch of size {batch_size}. ' +
                    f'Training will occur without any samples from this stream!')

        batches_per_stream.append(per_node_batches_per_stream)
        batches_from_partitions.append(per_node_stream_partitions)

    # Combine all device batches from all streams into one array, per node.
    all_partition_batches = []
    for node in range(world.num_nodes):
        all_partition_batches.append(np.concatenate(batches_from_partitions[node]))

    # Find the maximum number of device batches per node, for padding purposes.
    max_device_batches_per_node = max(
        [node_batches.shape[0] for node_batches in all_partition_batches])
    # If the maximum number of device batches per node is not divisible by the number of devices, increase
    # it so that it is. This is to ensure that later, we can reshape the device batches to global batches.
    num_devices = world.num_nodes * world.ranks_per_node
    padding_max_device_batches = num_devices - (max_device_batches_per_node % num_devices)
    max_device_batches_per_node += padding_max_device_batches

    # Shuffle seed changes with every epoch, so the order of streams in our batches also changes.
    epoch_rng = np.random.default_rng(dataset.shuffle_seed + epoch)

    # Shuffle the device batch origin order for each node.
    for node in range(world.num_nodes):
        # stream_origins is an array that tells us which stream each device batch is using.
        stream_origins = np.concatenate(
            [np.full(n_batch, i) for i, n_batch in enumerate(batches_per_stream[node])])
        epoch_rng.shuffle(stream_origins)

        # Now, we want the batch_indices array to correctly index into the node's device batches
        # array according to stream_origins in order to get our final device batch order.
        # For each stream, we want to traverse its device batches in the same order as its original partition.
        batch_indices = np.zeros(stream_origins.shape[0]).astype(np.int64)
        batch_offset = 0
        for i, n_device_batch in enumerate(batches_per_stream[node]):
            # Update batch_indices for the one stream at a time.
            batch_indices[stream_origins == i] += batch_offset + np.arange(n_device_batch)
            batch_offset += n_device_batch

        # Rearrange the node's device batches array by the batch_indices we have obtained.
        # This is a (num_device_batches, device_batch_size) array.
        all_partition_batches[node] = all_partition_batches[node][batch_indices]

        # If needed, pad the node's device batches array to have the same number of device batches
        # as the maximum number of device batches across all nodes.
        padding_batches = max_device_batches_per_node - all_partition_batches[node].shape[0]
        all_partition_batches[node] = np.concatenate(
            (all_partition_batches[node], np.full((padding_batches, batch_size), -1)))

    # Concatenate all the per-node device batches into one large sample partition across all nodes.
    # This sample partition is all device batches that are seen in the same order as in training.
    # So for example, we go from node 0: [a1, a2, a3], node 1: [b1, b2, b3], node 2: [c1, c2, c3]
    # to [a1, b1, c1, a2, b2, c2, a3, b3, c3], where each entry is a device batch.
    all_partition_batches = np.stack(all_partition_batches, axis=1).reshape(-1, batch_size)

    # If applicable we resume right after the most recently used full global batch.
    global_batch_size = batch_size * world.num_nodes * world.ranks_per_node
    if sample_in_epoch % global_batch_size != 0:
        logger.warning(
            'Because of the `device_per_stream` batching method, resumption may only occur on a sample \
            that is a multiple of the current global batch size of ' + str(global_batch_size) +
            '. Resuming training after the most recently finished global batch.')

    # Pad and reshape all_partition_batches to the global batch size instead of device batch size.
    # padding_batches = (global_batch_size // batch_size) - (all_partition_batches.shape[0] %
    #                                                     (global_batch_size // batch_size))
    # all_partition_batches = np.concatenate(
    #     (all_partition_batches, np.full((padding_batches, batch_size), -1)))
    # Padding to global_batch_size *should* be taken care of by the padding we did earlier, for node device batches.
    all_partition_batches = all_partition_batches.reshape(-1, global_batch_size)

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
