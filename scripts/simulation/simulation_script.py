# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Script for simulating streaming and displaying results."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulation_funcs import plot_simulation, simulate

# Input Parameters

# dataset
shards = 20000  # number of shards
samples_per_shard = 4093  # number of samples per shard
avg_shard_size = 15962700  # average shard size (bytes)

# training
epochs = 1
batches_per_epoch = 5000
device_batch_size = 4  # device batch size (samples)
avg_batch_time = 0.27  # average batch processing time (seconds)

# streaming
workers = 8  # number of workers per device
canonical_nodes = 1  # number of canonical nodes
predownload = 16  # number of samples to predownload per worker (samples)
cache_limit = 399067500  # cache limit per node (bytes)
shuffle_algo = 'py1b'  # shuffling algorithm
shuffle_block_size = 102325  # shuffling block size (samples)
seed = 17  # random seed

# hardware and network
physical_nodes = 1  # number of physical nodes
devices = 8  # number of devices per node
node_network_bandwidth = 1e8  # network bandwidth per node (bytes/s)

# ---------------------------------------------- #

# simulate step times and shard downloads given the inputs
step_times, shard_downloads = simulate(shards, samples_per_shard, avg_shard_size,
                                       device_batch_size, avg_batch_time, batches_per_epoch,
                                       epochs, physical_nodes, devices, node_network_bandwidth,
                                       workers, canonical_nodes, predownload, cache_limit,
                                       shuffle_algo, shuffle_block_size, seed)

# plot results
plot_simulation(step_times, shard_downloads, web=False)
