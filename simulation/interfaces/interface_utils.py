# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Peripheral functions for interface functionality."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Optional

import numpy as np
from core.utils import get_rolling_avg_throughput
from numpy.typing import NDArray

from streaming.base.util import number_abbrev_to_int


def plot_simulation(step_times: NDArray, step_downloads: NDArray, window: int = 10):
    """Plots simulation results for web UI or local script.

    Args:
        step_times (NDArray): time per step, as calculated by simulation
        step_downloads (NDArray): download size (bytes) per step, as calculated by simulation
        window (int, optional): window size to calculate batch throughput over. Defaults to ``10``.
    """
    import matplotlib.pyplot as plt

    immediate_batch_throughput = 1 / step_times

    step_downloads_cumulative = np.cumsum(step_downloads)

    batch_throughput_rolling_avg = get_rolling_avg_throughput(step_times, window)

    # matplotlib plot with 2 vertically stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    plt.suptitle('Simulation Results', fontsize=16)

    ax1.plot(np.arange(immediate_batch_throughput.shape[0]),
             immediate_batch_throughput,
             color='lightblue',
             label='per step throughput')
    ax1.plot(np.arange(batch_throughput_rolling_avg.shape[0]),
             batch_throughput_rolling_avg,
             color='darkblue',
             label='rolling throughput (10 step avg)')
    ax1.legend()
    ax1.set_ylim([0, max(immediate_batch_throughput) * 1.1])
    ax1.set_ylabel('batches/s')
    ax1.set_title('batch throughput (batches/s)')

    ax2.plot(np.arange(step_downloads_cumulative.shape[0]),
             step_downloads_cumulative,
             color='blue',
             label='total')
    ax2.set_ylim([0, max(step_downloads_cumulative) * 1.1])
    ax2.set_xlabel('step')
    ax2.set_ylabel('cumulative download (bytes)')
    ax2.set_title('network traffic (bytes)')

    fig.set_figheight(8)
    fig.set_figwidth(6)

    plt.show()


def get_train_dataset_params(input_params: dict, old_params: Optional[dict] = None) -> dict:
    """Get train dataset params from input params.

    Args:
        input_params (dict): The input parameter dictionary set by the user.
        old_params (Optional[dict], optional): Old parameters that may have been read in.
            Defaults to ``None``.

    Returns:
        dict: The full train dataset parameters.
    """
    train_dataset_params = {}
    train_dataset_params['epoch_size'] = input_params['epoch_size']
    train_dataset_params['batch_size'] = input_params['device_batch_size']
    train_dataset_params['nodes'] = input_params['physical_nodes']
    train_dataset_params['devices'] = input_params['devices']
    train_dataset_params['workers'] = input_params['workers']
    train_dataset_params['num_canonical_nodes'] = input_params['canonical_nodes']
    train_dataset_params['predownload'] = input_params['predownload']
    train_dataset_params['cache_limit'] = input_params['cache_limit']
    train_dataset_params['shuffle'] = input_params['shuffle']
    train_dataset_params['shuffle_algo'] = input_params['shuffle_algo']
    train_dataset_params['shuffle_block_size'] = number_abbrev_to_int(
        input_params['shuffle_block_size']) if input_params['shuffle_block_size'] is not None \
        else None
    train_dataset_params['shuffle_seed'] = input_params['seed']
    train_dataset_params['sampling_method'] = input_params['sampling_method']
    train_dataset_params['sampling_granularity'] = input_params['sampling_granularity']
    train_dataset_params['batching_method'] = input_params['batching_method']

    # If there were old params, fill them in.
    if old_params is not None:
        existing_params_set = set(train_dataset_params.keys())
        old_params_set = set(old_params.keys())
        # Keep params that were set in yaml but not accessible by the user in the UI.
        # This includes the old_params "local"/"remote" or "streams".
        for param in old_params_set - existing_params_set:
            train_dataset_params[param] = old_params[param]
    else:
        # If there are no old params, we need to set streams to what the user provided.
        train_dataset_params['streams'] = input_params['streams']

    return train_dataset_params
