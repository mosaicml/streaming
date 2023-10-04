# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Determine shuffle quality of a run over a fixed number of samples."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.utils import remove_padded_samples
from numpy.typing import NDArray

from streaming.base.partition.orig import get_partitions_orig
from streaming.base.shuffle import get_shuffle


def get_entropy(ordering: NDArray) -> float:
    """Calculate the entropy of an ordering, which is initially assumed to be in ascending order.

    Args:
        ordering (NDArray): The ordering to calculate the entropy of.

    Returns:
        float: The entropy of the ordering.
    """
    # get differences between elements
    diffs = np.diff(ordering)
    # diffs = np.insert(diffs, ordering.shape[0]-1, ordering[0]-ordering[-1])

    # change negative differences to positive
    diffs = np.abs(diffs)

    # count frequencies of differences
    diff_freqs = np.bincount(diffs)

    # remove zero frequency elements d
    diff_freqs = diff_freqs[diff_freqs != 0]

    # convert frequencies to probabilities
    diff_probs = diff_freqs / (ordering.shape[0] - 1)

    # calculate entropy
    diff_entropy = -np.sum(diff_probs * np.log2(diff_probs))

    return float(diff_entropy)


def get_partition_shard_info(epoch_size: int, canonical_nodes: int, physical_nodes: int,
                             devices: int, workers: int, device_batch_size: int,
                             samples_per_shard: int) -> tuple[NDArray, NDArray, NDArray]:
    """Partition up to 100 million samples and get associated shard information.

    Args:
        epoch_size (int): The number of samples in an epoch.
        canonical_nodes (int): The number of canonical nodes.
        physical_nodes (int): The number of physical nodes.
        devices (int): The number of devices.
        workers (int): The number of workers.
        device_batch_size (int): The batch size per device.
        samples_per_shard (int): Average number of samples per shard.

    Returns:
        tuple[NDArray, NDArray, NDArray]: The partition, in order, the
            sizes of each shard, and the mapping of sample id to shard id.
    """
    num_samples = epoch_size
    if num_samples > 100000000:
        print('Epoch size is >100 million. Using 100 million samples to analyze shuffle quality.')
        num_samples = 100000000

    partition = get_partitions_orig(num_samples, canonical_nodes, physical_nodes, devices, workers,
                                    device_batch_size)
    partition = partition.transpose(3, 2, 0, 1, 4).flatten()
    partition = remove_padded_samples(partition)

    # Construct shard sizes array.
    num_shards = num_samples // samples_per_shard
    shard_sizes = np.array([samples_per_shard] * num_shards)
    if num_samples % samples_per_shard != 0:
        num_shards += 1
        shard_sizes = np.append(shard_sizes, num_samples % samples_per_shard)

    # Construct sample id -> shard id mapping.
    shard_per_sample = np.repeat(np.arange(num_shards - 1), samples_per_shard)
    remaining_samples = num_samples - len(shard_per_sample)
    shard_per_sample = np.append(shard_per_sample, np.full(remaining_samples, num_shards - 1))

    return partition, shard_sizes, shard_per_sample


def get_entropy_shuffle_quality(shuffle_algo: str, partition: NDArray, shard_sizes: NDArray,
                                shard_per_sample: NDArray, canonical_nodes: int, seed: int,
                                shuffle_block_size: int) -> float:
    """Get the entropy of a shuffle, assuming samples and shards were initially in ascending order.

    Args:
        shuffle_algo (str): The shuffle algorithm to use.
        partition  (NDArray): The flattened, in-order partition to use.
        shard_sizes (NDArray): The sizes of each shard.
        shard_per_sample (NDArray): The mapping of sample id to shard id.
        canonical_nodes (int): The number of canonical nodes.
        seed (int): The seed to use for the shuffle.
        shuffle_block_size (int): The shuffle block size.

    Returns:
        float: The entropy of the shuffle, combining entropy from sample and shard orderings.
    """
    if shuffle_algo != 'none':
        # Assume we are shuffling only for epoch 0.
        shuffle_ordering = get_shuffle(shuffle_algo, shard_sizes, canonical_nodes, seed, 0,
                                       shuffle_block_size)
        partition = shuffle_ordering[partition]
    sample_entropy = get_entropy(partition)
    shard_entropy = get_entropy(shard_per_sample[partition])
    return sample_entropy + shard_entropy


def analyze_all_shuffle_quality(algos: list[str], canonical_nodes: int, physical_nodes: int,
                                devices: int, workers: int, device_batch_size: int,
                                shuffle_block_size: int, samples_per_shard: int, epoch_size: int,
                                seed: int) -> list[tuple[str, float]]:
    """Analyze the quality of this shuffle across algorithms.

    Args:
        algos (list[str]): The algorithms to analyze.
        canonical_nodes (int): The number of canonical nodes.
        physical_nodes (int): The number of physical nodes.
        devices (int): The number of devices.
        workers (int): The number of workers.
        device_batch_size (int): The batch size per device.
        shuffle_block_size (int): The shuffle block size.
        samples_per_shard (int): Average number of samples per shard.
        epoch_size (int): The number of samples in an epoch.
        seed (int): The seed to use for the shuffle.

    Returns:
        list[tuple[str, float]]: Shuffle algorithms and shuffle qualities.
    """
    print('Analyzing shuffle quality...')

    shuffle_qualities = []

    # Getting partition, shard_sizes, and shard_per_sample only has to be done once for all algos.
    partition, shard_sizes, shard_per_sample = get_partition_shard_info(
        epoch_size, canonical_nodes, physical_nodes, devices, workers, device_batch_size,
        samples_per_shard)
    for algo in algos:
        shuffle_qualities.append(
            get_entropy_shuffle_quality(algo, partition, shard_sizes, shard_per_sample,
                                        canonical_nodes, seed, shuffle_block_size))

    return shuffle_qualities


def analyze_shuffle_quality(algo: str, canonical_nodes: int, physical_nodes: int, devices: int,
                            workers: int, device_batch_size: int, shuffle_block_size: int,
                            samples_per_shard: int, epoch_size: int,
                            seed: int) -> tuple[str, float]:
    """Analyze the quality of a shuffle for one algorithm.

    Args:
        algo (str): The algorithm to analyze.
        canonical_nodes (int): The number of canonical nodes.
        physical_nodes (int): The number of physical nodes.
        devices (int): The number of devices.
        workers (int): The number of workers.
        device_batch_size (int): The batch size per device.
        shuffle_block_size (int): The shuffle block size.
        samples_per_shard (int): Average number of samples per shard.
        epoch_size (int): The number of samples in an epoch.
        seed (int): The seed to use for the shuffle.

    Returns:
        tuple[str, float]: Shuffle algorithm and shuffle quality.
    """
    print(f'Analyzing shuffle quality for {algo}...')

    # Getting partition, shard_sizes, and shard_per_sample only has to be done once for all algos.
    partition, shard_sizes, shard_per_sample = get_partition_shard_info(
        epoch_size, canonical_nodes, physical_nodes, devices, workers, device_batch_size,
        samples_per_shard)

    shuffle_quality = get_entropy_shuffle_quality(algo, partition, shard_sizes, shard_per_sample,
                                                  canonical_nodes, seed, shuffle_block_size)

    return algo, shuffle_quality
