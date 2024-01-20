# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Peripheral functions for simulation functionality."""

from typing import Tuple

import numpy as np
from core.sim_dataset import SimulationDataset
from core.sim_time import Time, TimeUnit
from numpy.typing import NDArray


def get_batches_epochs(dataset: SimulationDataset, max_duration: Time) -> Tuple[int, int, int]:
    """Get batches per epoch, epochs, and total epochs from a Time object.

    Args:
        dataset (SimulationDataset): The dataset being simulated.
        max_duration (Time): The maximum duration, can be specified in yaml.

    Returns:
        Tuple[int, int, int]: batches per epoch, epochs, and the total batches.
    """
    # get epochs, batches_per_epoch, and total_batches from a Time obect
    dataset_batches = dataset.get_num_batches()
    batches_per_epoch = dataset_batches
    epochs = 1
    total_batches = dataset_batches
    if max_duration.unit == TimeUnit.EPOCH:
        epochs = max_duration.value
        batches_per_epoch = dataset_batches
        total_batches = epochs * batches_per_epoch
    elif max_duration.unit == TimeUnit.BATCH:
        full_epochs = max_duration.value // dataset_batches
        # check if there is a partial epoch we should fulfill
        if max_duration.value % dataset_batches != 0:
            full_epochs += 1
        # make sure we don't simulate past the duration set.
        if max_duration.value < dataset_batches:
            batches_per_epoch = max_duration.value
        else:
            batches_per_epoch = dataset_batches
        total_batches = max_duration.value
    else:
        raise ValueError('Simulator currently only supports max_duration in epochs or batches.')

    return batches_per_epoch, epochs, total_batches


def get_total_batches(dataset: SimulationDataset, max_duration: Time) -> int:
    """Get total batches from a Time object.

    Args:
        dataset (SimulationDataset): The dataset being simulated.
        max_duration (Time): The maximum duration, can be specified in yaml.

    Returns:
        int: The total batches.
    """
    dataset_batches = dataset.get_num_batches()
    total_batches = dataset_batches
    if max_duration.unit == TimeUnit.EPOCH:
        epochs = max_duration.value
        batches_per_epoch = dataset_batches
        total_batches = epochs * batches_per_epoch
    elif max_duration.unit == TimeUnit.BATCH:
        total_batches = max_duration.value
    else:
        raise ValueError('Simulator currently only supports max_duration in epochs or batches.')

    return total_batches


def remove_padded_samples(samples: NDArray) -> NDArray:
    """Remove padded samples from a batch.

    Args:
        samples (NDArray): The samples to remove padded samples from.

    Returns:
        NDArray: The samples with padded samples removed.
    """
    return np.delete(samples, np.where(samples == -1))


def bytes_to_time(bytes: int, bandwidth: int) -> float:
    """Convert bytes to time.

    Args:
        bytes (int): The bytes to convert.
        bandwidth (int): The bandwidth available.

    Returns:
        float: The time it takes to transfer the bytes.
    """
    return bytes / bandwidth


def time_to_bytes(time: float, bandwidth: int) -> int:
    """Convert time to bytes.

    Args:
        time (float): The time to convert.
        bandwidth (int): The bandwidth available.

    Returns:
        int: The bytes transferred in the time.
    """
    return int(time * bandwidth)


def get_rolling_avg_throughput(step_times: NDArray, window: int = 10) -> NDArray:
    """Get rolling average throughput from step times.

    Args:
        step_times (NDArray): time per step, as calculated by simulation
        window (int): window size for rolling average

    Returns:
        NDArray: rolling average throughput
    """
    step_times_rolling_avg = np.convolve(step_times, np.ones(window) / window, mode='valid')
    batch_throughput_rolling_avg = 1 / step_times_rolling_avg
    batch_throughput_rolling_avg = np.concatenate(
        (np.array([0] * (window - 1)), batch_throughput_rolling_avg))

    return batch_throughput_rolling_avg


def get_simulation_stats(step_times: NDArray, time_per_sample: float,
                         device_batch_size: int) -> Tuple[int, float, int, int]:
    """Gets simulation stats for web UI.

    Args:
        step_times (NDArray): time per step, as calculated by simulation
        time_per_sample (float): time to process one sample on one device (seconds)
        device_batch_size (int): batch size per device

    Returns:
        Tuple[int, float, int, int]: number of steps with throughput drops, time till warmup,
            step number of warmup, number of steps with throughput drops after warmup
    """
    # calculate percent of download-limited steps
    min_step_time = time_per_sample * device_batch_size
    all_throughput_drops = int(np.count_nonzero(step_times > (min_step_time)))

    epsilon = 1e-6

    # calculate warmup time (time to first max possible rolling average throughput) within epsilon
    max_throughput = 1 / min_step_time
    rolling_avg_throughput = get_rolling_avg_throughput(step_times)
    if np.max(rolling_avg_throughput) >= max_throughput - epsilon:
        warmup_step = int(np.argmax(rolling_avg_throughput >= (max_throughput)) + 1)
        warmup_time = float(np.sum(step_times[:warmup_step]))
    else:
        # we never hit the max possible throughput
        warmup_step = int(rolling_avg_throughput.shape[0])
        warmup_time = float(np.sum(step_times))

    # see if there are throughput drops after warmup so we can notify users
    if warmup_step != rolling_avg_throughput.shape[0]:
        # if we did hit the max throughput then we check for later drops
        post_warmup_tp_drops = int(np.count_nonzero(step_times[warmup_step:] > min_step_time))
    else:
        # since warmup was the whole time, there are no post-warmup throughput drops
        post_warmup_tp_drops = 0

    return all_throughput_drops, warmup_time, warmup_step, post_warmup_tp_drops
