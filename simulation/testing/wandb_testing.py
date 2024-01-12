# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Test simulation results against run results from a wandb project."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from core.create_index import create_stream_index
from core.main import simulate
from core.sim_dataset import SimulationDataset
from core.sim_time import TimeUnit, ensure_time
from numpy.typing import NDArray

from streaming.base import Stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

api = wandb.Api()

project_id = 'mosaic-ml/streaming-shuffling-algo'
project_runs = api.runs(path=project_id, per_page=300)
project_runs_list = [run.id for run in project_runs]
skip = 0

# Enter the dataset parameters here.
# These are the params for C4 dataset, gpt-neox tokenized.
shards = 20850
samples_per_shard = 4093
avg_raw_shard_size = 67092639
avg_zip_shard_size = 16000000
time_per_sample = 0.0175
node_network_bandwidth = 1e9
throughput_window = 10


def get_similarity_percentage(real: NDArray, sim: NDArray) -> float:
    """Get similarity score between real and simulated data.

    Args:
        real (NDArray): The real data.
        sim (NDArray): The simulated data.

    Returns:
        float: The similarity score, between 0 and 1.
    """
    real_copy = real.reshape(1, -1)
    sim_copy = sim.reshape(1, -1)
    merged = np.concatenate((real_copy, sim_copy), axis=0)
    similarities = np.abs(real - sim) / np.max(merged, axis=0)
    nanmean = np.nanmean(similarities)
    return float(1 - nanmean)


for run_id in project_runs_list[skip:]:

    run = api.run(f'{project_id}/{run_id}')

    print(run.name)

    summary = run.summary
    config = run.config

    if '_step' not in summary:
        logger.warning(' Skipping unsuccessful run.')
        continue

    # get parameters from run config and summary
    max_duration_value = summary['_step']
    max_duration = ensure_time(str(max_duration_value) + 'ba', TimeUnit.EPOCH)
    devices = int(config['num_gpus_per_node'])
    physical_nodes = int(config['n_gpus'] / devices)
    # device_batch_size set for each run
    device_batch_size = int(config['global_train_batch_size'] / (physical_nodes * devices))
    canonical_nodes = int(config['num_canonical_nodes'])
    workers = int(config['train_loader']['num_workers'])
    predownload = int(config['train_loader']['dataset']['predownload'])
    cache_limit = None
    if 'cache_limit' in config['train_loader']['dataset']:
        cache_limit = config['train_loader']['dataset']['cache_limit']
    shuffle = True
    if 'shuffle' in config['train_loader']['dataset']:
        shuffle = config['train_loader']['dataset']['shuffle']
    shuffle_algo = 'py1e'
    if 'shuffle_algo' in config['train_loader']['dataset']:
        shuffle_algo = str(config['train_loader']['dataset']['shuffle_algo'])
    shuffle_block_size = config['train_loader']['dataset']['shuffle_block_size']
    seed = config['seed']

    # get step timestamps, real throughput, and network use from the run
    step_timestamps = run.history(samples=max_duration_value, keys=['_timestamp'], pandas=True)
    real_batch_throughput = run.history(samples=max_duration_value - throughput_window,
                                        keys=['throughput/batches_per_sec'],
                                        pandas=True)

    real_network_use = run.history(stream='system',
                                   pandas=True)[['_timestamp', 'system.network.recv']]

    # merge real_network_use with step_timestamps
    merged_network_use = pd.merge_asof(real_network_use,
                                       step_timestamps,
                                       on='_timestamp',
                                       direction='nearest')

    # simulate throughput and network use given the inputs
    stream_indexpath = create_stream_index(shards, samples_per_shard, avg_raw_shard_size,
                                           avg_zip_shard_size)
    stream_folder = os.path.dirname(stream_indexpath)
    stream = Stream(local=stream_folder)

    dataset = SimulationDataset(nodes=physical_nodes,
                                devices=devices,
                                workers=workers,
                                streams=[stream],
                                predownload=predownload,
                                cache_limit=cache_limit,
                                num_canonical_nodes=canonical_nodes,
                                batch_size=device_batch_size,
                                shuffle=shuffle,
                                shuffle_algo=shuffle_algo,
                                shuffle_seed=seed,
                                shuffle_block_size=shuffle_block_size)

    node_network_bandwidth = int(node_network_bandwidth)
    results = next(
        simulate(dataset=dataset,
                 time_per_sample=time_per_sample,
                 node_network_bandwidth=node_network_bandwidth,
                 max_duration=max_duration))

    if len(results) != 4:
        raise ValueError(f'Simulation with generate=False should return 4 final results. ' +
                         f'Instead, received `results` of length {len(results)}.')
    step_times, step_downloads, startup_time, min_cache_limit = results

    immediate_batch_throughput = 1 / step_times

    shard_downloads_cumulative = np.cumsum(step_downloads)
    shard_downloads_steps = np.arange(step_downloads.shape[0])
    sim_downloads = pd.DataFrame({
        '_step': shard_downloads_steps,
        'sim_downloads': shard_downloads_cumulative
    })
    # merge simulated downloads with real downloads dataframe
    merged_network_use = pd.merge_asof(merged_network_use,
                                       sim_downloads,
                                       on='_step',
                                       direction='nearest')

    step_times_rolling_avg = np.convolve(step_times,
                                         np.ones(throughput_window) / throughput_window,
                                         mode='valid')[:-1]
    batch_throughput_rolling_avg = 1 / step_times_rolling_avg
    sim_throughput = pd.DataFrame({'_step': throughput_window + \
                                   np.arange(batch_throughput_rolling_avg.shape[0]),
                                     'sim_throughput': batch_throughput_rolling_avg})
    merged_throughput = pd.merge_asof(real_batch_throughput,
                                      sim_throughput,
                                      on='_step',
                                      direction='nearest')

    # get similarity scores
    throughput_similarity = get_similarity_percentage(
        merged_throughput['throughput/batches_per_sec'].to_numpy(),
        merged_throughput['sim_throughput'].to_numpy())
    network_similarity = get_similarity_percentage(
        physical_nodes * (merged_network_use['system.network.recv'].to_numpy()),
        (merged_network_use['sim_downloads'].to_numpy()))

    # log params and results to easily paste to spreadsheet
    print(run.name, seed, canonical_nodes, physical_nodes, predownload, shuffle_algo,
          shuffle_block_size, cache_limit, max_duration_value, throughput_similarity,
          network_similarity)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title('throughput - score: ' + str(throughput_similarity))
    ax1.plot(merged_throughput['_step'],
             merged_throughput['throughput/batches_per_sec'],
             color='red',
             label='real')
    ax1.plot(merged_throughput['_step'],
             merged_throughput['sim_throughput'],
             color='blue',
             label='sim')
    ax1.legend()

    ax2.set_title('network use - score: ' + str(network_similarity))
    # wandb only logs network use for node 0. multiply by number of nodes to get total network use
    ax2.plot(merged_network_use['_timestamp'],
             physical_nodes * merged_network_use['system.network.recv'],
             color='red',
             label='real')
    # simulation assumes all shards are downloaded uncompressed (overestimates).
    ax2.plot(merged_network_use['_timestamp'],
             merged_network_use['sim_downloads'],
             color='blue',
             label='sim')
    ax2.legend()

    fig.set_figheight(8)

    plt.show()
