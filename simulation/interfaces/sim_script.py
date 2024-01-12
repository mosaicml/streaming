# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Script for simulating training downloads and throughput, and displaying results."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import humanize
from core.create_index import create_stream_index
from core.main import simulate
from core.sim_dataset import SimulationDataset
from core.sim_time import TimeUnit, ensure_time
from core.utils import get_simulation_stats
from interfaces.interface_utils import plot_simulation

from streaming.base import Stream

# Input Parameters

# dataset
shards = 20850  # number of shards
samples_per_shard = 4093  # number of samples per shard
avg_raw_shard_size = 67092639  # average shard size (bytes)
avg_zip_shard_size = 15000000  # average compressed shard size (bytes)

# training
max_duration = '1000ba'  # max duration of training (batches: "ba", epochs: "ep")
epoch_size = None  # epoch size (samples)
device_batch_size = 16  # device batch size (samples)

# streaming
workers = 8  # number of workers per device
canonical_nodes = 2  # number of canonical nodes
predownload = 32  # number of samples to predownload per worker (samples)
cache_limit = None  # cache limit (bytes)
shuffle = True  # whether to shuffle dataset
shuffle_algo = 'py1b'  # shuffling algorithm
shuffle_block_size = 16000000  # shuffling block size (samples)
seed = 17  # random seed

# hardware and network
physical_nodes = 2  # number of physical nodes
devices = 8  # number of devices per node
time_per_sample = 0.0175  # time to process one sample on one device (seconds)
node_internet_bandwidth = 1e7  # network internet per node (bytes/s)

# ---------------------------------------------- #

# instantiate SimulationDataset on the same parameters for the new simulation function

stream_indexpath = create_stream_index(shards, samples_per_shard, avg_raw_shard_size,
                                       avg_zip_shard_size)
stream_folder = os.path.dirname(stream_indexpath)
stream = Stream(local=stream_folder)
max_duration = ensure_time(max_duration, TimeUnit.EPOCH)

dataset = SimulationDataset(nodes=physical_nodes,
                            devices=devices,
                            workers=workers,
                            streams=[stream],
                            epoch_size=epoch_size,
                            predownload=predownload,
                            cache_limit=cache_limit,
                            num_canonical_nodes=canonical_nodes,
                            batch_size=device_batch_size,
                            shuffle=True,
                            shuffle_algo=shuffle_algo,
                            shuffle_seed=seed,
                            shuffle_block_size=shuffle_block_size)

node_internet_bandwidth = int(node_internet_bandwidth)
results = next(
    simulate(dataset=dataset,
             time_per_sample=time_per_sample,
             node_network_bandwidth=node_internet_bandwidth,
             max_duration=max_duration))

if len(results) != 4:
    raise ValueError(f'Simulation with generate=False should return 4 final results. ' +
                     f'Instead, received `results` of length {len(results)}.')
step_times, step_downloads, startup_time, min_cache_limit = results
global_batch_size = device_batch_size * devices * physical_nodes

# Display simulation stats
total_batches = len(step_times)
all_throughput_drops, warmup_time, warmup_step, post_warmup_throughput_drops = \
    get_simulation_stats(step_times, time_per_sample, global_batch_size//(physical_nodes*devices))
print('\nSimulation Stats:')
print(f'Minimum cache limit needed: {humanize.naturalsize(min_cache_limit)}')
if cache_limit is not None and cache_limit < min_cache_limit:
    # Cache limit is too low, and will cause shard redownloads / throughput drops.
    print('⚠️ The provided cache limit is lower than the minimum cache limit needed to \
          prevent shard re-downloads. This can cause throughput issues.')
if warmup_step == total_batches:
    # display error if the warmup phase is the whole run, so we never hit peak throughput.
    print('🚨 This configuration is severely bottlenecked by downloading. The run will not be \
            performant.')
elif post_warmup_throughput_drops:
    # display warning if post-warmup throughput drops are more than 10% of the run.
    print(
        '⚠️ This configuration experiences some downloading-related slowdowns even after warmup.')
print('{0} steps, or {1:.1f}% of all steps, waited for shard downloads.'\
      .format(all_throughput_drops, 100 * all_throughput_drops / (total_batches)))
if warmup_step != total_batches:
    # only display post-warmup throughput drop info if we actually ended the warmup period (i.e. we hit peak throughput at some point)
    print('There were {} steps that waited for shard downloads after the warmup period.'\
          .format(post_warmup_throughput_drops))
print('Estimated time to first batch: {0:.2f} s'.format(startup_time))
print('Estimated warmup time: {0:.2f} s'.format(warmup_time))

plot_simulation(step_times, step_downloads)
