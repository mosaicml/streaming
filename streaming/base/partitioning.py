# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Partitions the sample space to nodes, devices, and workers."""

import math

import numpy as np


def get_partitions_slow(dataset_size: int,
                        num_canonical_nodes: int,
                        num_physical_nodes: int,
                        ranks_per_node: int,
                        workers_per_rank: int,
                        device_batch_size: int = 1,
                        drop_first: int = 0):
    """Partition the given number of samples to nodes, devices, and workers.

    Args:
        dataset_size (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        device_batch_size (int): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``1``.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes x ranks x workers x samples).
    """
    # Divide the full dataset sample range into sample range per canonical node.
    device_batch_size = device_batch_size or 1
    arrs = []
    for i in range(num_canonical_nodes):
        start = i * dataset_size // num_canonical_nodes
        stop = (i + 1) * dataset_size // num_canonical_nodes
        arr = np.arange(start, stop)
        arrs.append(arr)

    # Make the spans equal length by repeating the last sample if too short.
    # -> Shape: (canonical nodes x samples).
    max_len = max(map(len, arrs))
    for i, x in enumerate(arrs):
        if len(x) < max_len:
            last = np.array([x[-1]])
            arrs[i] = np.concatenate([x, last])
    x = np.stack(arrs)

    # If there are more physical than canonical nodes, interleave canonical onto physical nodes.
    # -> Shape: (canonical nodes x samples).
    if num_canonical_nodes < num_physical_nodes:
        assert not num_physical_nodes % num_canonical_nodes
        ratio = num_physical_nodes // num_canonical_nodes
        too_many = x.shape[1] % ratio
        if too_many:
            too_few = ratio - too_many
            last = x[:, -ratio - too_few + 1:-ratio + 1]
            x = np.concatenate([x, last], 1)

    # Drop samples that have already been seen and reshape.
    # -> Shape: (physical nodes x samples).
    x = x.transpose()
    x = x.flatten()
    x = x[drop_first:]
    x = x.reshape(-1, num_physical_nodes)
    x = x.transpose()

    # Interleave the node sample ranges over each node's devices, padding by repeating last sample.
    # -> Shape: (physical nodes x samples x devices).
    too_many = x.shape[1] % ranks_per_node
    if too_many:
        too_few = ranks_per_node - too_many
        last = x[:, -ranks_per_node - too_few + 1:-ranks_per_node + 1]
        x = np.concatenate([x, last], 1)
    x = x.reshape(num_physical_nodes, -1, ranks_per_node)

    # Interleave each device's samples across its workers, padding with -1.
    # -> Shape: (physical nodes x samples x workers x devices).
    too_many = x.shape[1] % workers_per_rank
    # Make the number of samples multiple of batch size and workers per rank
    rounded_num_samples = math.ceil(x.shape[1] / (workers_per_rank * device_batch_size)) * \
        (workers_per_rank * device_batch_size)
    too_many = rounded_num_samples - x.shape[1]
    if too_many:
        last = np.full((num_physical_nodes, too_many, ranks_per_node), -1, np.int64)
        x = np.concatenate([x, last], 1)

    # -> Shape: (physical nodes x ranks per node x samples)
    x = x.transpose(0, 2, 1)
    x = x.reshape(num_physical_nodes, ranks_per_node, -1, device_batch_size)
    x = np.concatenate(
        [x[:, :, np.arange(i, x.shape[2], workers_per_rank), :] for i in range(workers_per_rank)],
        axis=2)

    # -> Shape: (physical nodes x ranks per node x workers per rank x samples)
    return x.reshape(num_physical_nodes, ranks_per_node, workers_per_rank, -1)


def get_partitions_fast(dataset_size: int,
                        num_canonical_nodes: int,
                        num_physical_nodes: int,
                        ranks_per_node: int,
                        workers_per_rank: int,
                        device_batch_size: int = 1,
                        drop_first: int = 0):
    """Partition the given number of samples to nodes, devices, and workers.

    Args:
        dataset_size (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        device_batch_size (int): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``1``.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes x ranks per node x workers per rank x batches per worker x device batch size).
    """
    if num_canonical_nodes % num_physical_nodes and num_physical_nodes % num_canonical_nodes:
        raise ValueError('One of {canonical nodes, physical nodes} must be evenly divisible by ' +
                         'the other.')
    device_batch_size = device_batch_size or 1

    devices = num_physical_nodes * ranks_per_node
    device_samples = math.ceil(dataset_size / devices)
    worker_batches = math.ceil(device_samples / (workers_per_rank * device_batch_size))
    padded_device_samples = workers_per_rank * worker_batches * device_batch_size

    node_starts = dataset_size * np.arange(num_canonical_nodes) // num_canonical_nodes
    per_node_device_starts = np.arange(ranks_per_node)
    step = ranks_per_node

    if num_canonical_nodes < num_physical_nodes:
        node_ratio = num_physical_nodes // num_canonical_nodes
        node_starts = np.tile(node_starts, node_ratio)
        node_starts += np.arange(node_ratio).repeat(num_canonical_nodes)
        per_node_device_starts *= node_ratio
        step *= node_ratio

    # starts shape -> (num nodes x ranks per node x 1)
    starts = node_starts.reshape(-1, 1, 1) + per_node_device_starts.reshape(1, -1, 1)
    # indices shape -> (1 x 1 x padded device samples)
    indices = np.arange(padded_device_samples).reshape(1, 1, -1)
    # ids shape -> (num_canonical_nodes x ranks per node x padded device samples)
    ids = starts + indices * step

    if num_physical_nodes < num_canonical_nodes:
        # reshape the ids so that the order of the samples is preserved based on the provided number
        # of canonical nodes
        node_ratio = num_canonical_nodes // num_physical_nodes
        ids = ids.reshape(node_ratio, num_physical_nodes, ranks_per_node, -1)
        ids = ids.transpose(1, 3, 2, 0)
        ids = ids.reshape(num_physical_nodes, -1, ranks_per_node)
        ids = ids.transpose(0, 2, 1)
        ids = ids[:, :, :padded_device_samples]

    # ids shape -> (physical nodes x ranks per node x padded device samples)
    second_to_last = ids[:, :, device_samples - 2]
    last = ids[:, :, device_samples - 1]
    # duplicates valid samples at the end, so that each device has the same number of samples
    ids[:, :, device_samples - 1] = np.where(last < dataset_size, last, second_to_last)
    # drops all samples past the end of the dataset
    ids[:, :, device_samples:] = -1

    if drop_first:
        # drop the first drop_first samples by flattening into the order that samples would be seen and
        # clipping the samples from the left
        ids = ids.transpose(2, 0, 1)
        ids = ids.flatten()
        ids[:-drop_first] = ids[drop_first:]
        ids[-drop_first:] = -1
        ids = ids.reshape(padded_device_samples, num_physical_nodes, ranks_per_node)
        # return to original ids shape of (physical nodes x ranks per node x padded device samples)
        ids = ids.transpose(1, 2, 0)

    ids = ids.reshape(num_physical_nodes, ranks_per_node, worker_batches, workers_per_rank,
                      device_batch_size)
    # ids shape -> (physical nodes x ranks per node x workers per rank x worker batches x device batch size)
    return ids.transpose(0, 1, 3, 2, 4)


get_partitions = get_partitions_fast
