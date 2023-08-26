# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Script for simulating streaming and displaying results."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulation_funcs import plot_simulation, simulate

# Input Parameters

# dataset
shards = 20850  # number of shards
samples_per_shard = 4093  # number of samples per shard
avg_shard_size = 67092639  # average shard size (bytes)

# training
epochs = 1
batches_per_epoch = 3000
device_batch_size = 16  # device batch size (samples)

# streaming
workers = 8  # number of workers per device
canonical_nodes = 128  # number of canonical nodes
predownload = 3800  # number of samples to predownload per worker (samples)
shuffle_algo = 'py1b'  # shuffling algorithm
cache_limit = None  # cache limit (bytes)
shuffle_block_size = 1000000  # shuffling block size (samples)
seed = 18  # random seed

# hardware and network
physical_nodes = 2  # number of physical nodes
devices = 8  # number of devices per node
time_per_sample = 0.0175  # time to process one sample on one device (seconds)
node_network_bandwidth = 2e9  # network bandwidth per node (bytes/s)

# ---------------------------------------------- #

# simulate step times and shard downloads given the inputs
results = simulate(shards, samples_per_shard, avg_shard_size,
                                       device_batch_size, time_per_sample, batches_per_epoch,
                                       epochs, physical_nodes, devices, node_network_bandwidth,
                                       workers, canonical_nodes, predownload, cache_limit, 
                                       shuffle_algo, shuffle_block_size, seed)

step_times, shard_downloads = next(results)

# plot results
plot_simulation(step_times, shard_downloads, web=False)
