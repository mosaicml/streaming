# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""sim_cli: simulate your training yaml from the command line."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

import humanize
from core.main import simulate
from core.utils import get_simulation_stats
from core.yaml_processing import create_simulation_dataset, ingest_yaml
from interfaces.interface_utils import plot_simulation

from streaming.base.util import bytes_to_int

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate your training yaml from the command \
                                     line.')
    parser.add_argument('-f', '--file', type=str, help='path to yaml file', required=True)
    parser.add_argument('-n', '--nodes', type=int, help='number of physical nodes', required=False)
    parser.add_argument('-d',
                        '--devices',
                        type=int,
                        help='number of devices per node',
                        required=False)
    parser.add_argument('-t',
                        '--time_per_sample',
                        type=float,
                        help='time to process one sample on one device (seconds)',
                        required=False)
    parser.add_argument('-b',
                        '--node_internet_bandwidth',
                        type=str,
                        help='internet bandwidth per node (bytes/s)',
                        required=False)
    args = parser.parse_args()

    # Read in yaml file
    filepath = args.file
    total_devices, workers, max_duration, global_batch_size, train_dataset = \
        ingest_yaml(filepath=filepath)

    # Check if we have to ask for any parameters
    args = parser.parse_args()
    nodes = args.nodes
    if nodes is None:
        nodes = int(input('Number of physical nodes: '))
    # devices may be specified in the yaml file.
    if total_devices is None:
        devices = args.devices
    else:
        if total_devices % nodes != 0:
            raise ValueError('The number of devices must be divisible by the number of nodes.')
        devices = total_devices // nodes
    time_per_sample = args.time_per_sample
    node_network_bandwidth = args.node_internet_bandwidth
    if devices is None:
        devices = int(input('Number of devices per node: '))
    if time_per_sample is None:
        time_per_sample = float(input('Time to process one sample on one device (seconds): '))
    if node_network_bandwidth is None:
        bandwidth_input = input('Internet bandwidth per node (bytes/s): ')
        try:
            # Converting to float first handles the case where the input is a string in scientific
            # notation, like "1e9".
            node_network_bandwidth = float(bandwidth_input)
            node_network_bandwidth = int(node_network_bandwidth)
        except ValueError:
            node_network_bandwidth = str(bandwidth_input)

    # Convert strings into numbers for applicable args
    node_network_bandwidth = bytes_to_int(node_network_bandwidth)

    # Create SimulationDataset
    print('Constructing SimulationDataset...')
    dataset = create_simulation_dataset(nodes, devices, workers, global_batch_size, train_dataset)

    # Simulate Run
    results = next(simulate(dataset, time_per_sample, node_network_bandwidth, max_duration))
    if len(results) != 4:
        raise ValueError(f'Simulation with generate=False should return 4 final results. ' +
                         f'Instead, received `results` of length {len(results)}.')
    step_times, step_downloads, startup_time, min_cache_limit = results

    print('Simulation Finished.')

    # Display simulation stats
    total_batches = len(step_times)
    cache_limit = dataset.get_cache_limit()
    all_throughput_drops, warmup_time, warmup_step, post_warmup_throughput_drops = \
        get_simulation_stats(step_times, time_per_sample, global_batch_size//(nodes*devices))
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
        print('⚠️ This configuration experiences some downloading-related slowdowns even after \
              warmup.')
    print('{0} steps, or {1:.1f}% of all steps, waited for shard downloads.'\
          .format(all_throughput_drops, 100 * all_throughput_drops / (total_batches)))
    if warmup_step != total_batches:
        # only display post-warmup throughput drop info if we actually ended the warmup period
        # (i.e. we hit peak throughput at some point)
        print('There were {} steps that waited for shard downloads after the warmup period.'\
              .format(post_warmup_throughput_drops))
    print('Estimated time to first batch: {0:.2f} s'.format(startup_time))
    print('Estimated warmup time: {0:.2f} s'.format(warmup_time))

    # Plot simulation
    plot_simulation(step_times, step_downloads)
