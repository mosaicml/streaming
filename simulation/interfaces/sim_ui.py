# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Simulator web UI using streamlit."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from typing import Union

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from core.create_index import create_stream_index
from core.main import simulate
from core.shuffle_quality import analyze_shuffle_quality_entropy
from core.sim_dataset import SimulationDataset
from core.sim_time import Time
from core.utils import get_total_batches
from core.yaml_processing import create_simulation_dataset, ingest_yaml
from interfaces.interface_utils import get_train_dataset_params
from interfaces.widgets import (display_shuffle_quality_graph, display_simulation_stats,
                                get_line_chart, param_inputs)

from streaming.base.util import bytes_to_int, number_abbrev_to_int

# set up page
st.set_page_config(layout='wide')
col1, space, col2 = st.columns((10, 1, 6))
col2.title('Streaming Simulator')
col2.write('Enter run parameters in the left panel.')
col2.text('')
progress_bar = col1.progress(0)
status_text = col1.empty()
col1.text('')
throughput_plot = col2.empty()
network_plot = col2.empty()
sim_stats = col2.empty()
col2.text('')
shuffle_quality_plot = col2.empty()
throughput_window = 10
shuffle_quality_algos = ['naive', 'py1b', 'py1br', 'py1e', 'py1s', 'py2s', 'none']


def submit_jobs(shuffle_quality: bool, dataset: SimulationDataset, time_per_sample: float,
                node_internet_bandwidth: Union[int, str], max_duration: Time):
    """Submit jobs to the executor for simulation.

    Args:
        shuffle_quality (bool): Whether to run shuffle quality analysis.
        dataset (SimulationDataset): Dataset to use for simulation.
        time_per_sample (float): Time for one device to process one sample.
        node_internet_bandwidth (Union[int,str]): Internet bandwidth per node.
        max_duration (Time): Maximum duration of simulation.
    """
    total_batches = get_total_batches(dataset=dataset, max_duration=max_duration)
    node_internet_bandwidth = bytes_to_int(node_internet_bandwidth)
    cache_limit = dataset.get_cache_limit()
    gen_sim = simulate(dataset,
                       time_per_sample,
                       node_internet_bandwidth,
                       max_duration,
                       generator=True)
    gen_step_times = []
    gen_step_downloads = []
    rolling_throughput_data = []
    immediate_throughput_data = []
    network_data = []
    steps = []
    time_to_first_batch = 0
    futures = []
    shuffle_quality_graphed = False
    # Initialize min_cache_limit to be 0. Will be replaced by the simulated value.
    min_cache_limit = 0
    # Define partial function to pass to executor map for simulation.
    with ProcessPoolExecutor() as executor:
        # Submit shuffle quality job to executor.
        if shuffle_quality:
            col1.write('Starting shuffle quality analysis...')
            input_params = st.session_state['input_params']
            # Use multiprocessing to get the shuffle quality results.
            canonical_nodes = input_params['canonical_nodes']
            if canonical_nodes is None:
                canonical_nodes = dataset.get_num_canonical_nodes()
            physical_nodes = input_params['physical_nodes']
            devices = input_params['devices']
            workers = input_params['workers']
            device_batch_size = input_params['device_batch_size']
            shuffle_block_size = number_abbrev_to_int(input_params['shuffle_block_size']) \
                if input_params['shuffle_block_size'] is not None \
                    else dataset.get_shuffle_block_size()
            samples_per_shard = dataset.get_avg_samples_per_shard()
            epoch_size = dataset.get_epoch_size()
            if epoch_size > 100_000_000:
                st.warning('Epoch size is over 100 million samples. Shuffle quality analysis \
                           will be conducted only on the first 100 million samples.',
                           icon='⚠️')
            seed = input_params['seed']
            # Submit all shuffle quality analysis jobs to executor.
            futures = [
                executor.submit(analyze_shuffle_quality_entropy, algo, canonical_nodes,
                                physical_nodes, devices, workers, device_batch_size,
                                shuffle_block_size, samples_per_shard, epoch_size, seed)
                for algo in shuffle_quality_algos
            ]

        # Simulate only on the main worker, otherwise it's super slow.
        for output in gen_sim:
            # If output is a length 2, it is the time to first batch and min cache limit.
            # Otherwise it is the step, step time, and shard download from the simulation.
            if len(output) == 2:
                step = total_batches - 1
                time_to_first_batch, min_cache_limit = output
            else:
                if len(output) != 3:
                    raise ValueError(f'Simulation with generate=True should return 3 results per' +
                                     f' step. Instead, output was length {len(output)}.')

                step, step_time, shard_download = output
                gen_step_times.append(step_time)
                gen_step_downloads.append(shard_download)
                # plot throughput once we have enough samples for the window
                rolling_throughput = 0
                if step >= throughput_window - 1:
                    step_time_window = np.array(gen_step_times[-throughput_window:])
                    rolling_throughput = 1 / np.mean((step_time_window))
                rolling_throughput_data.append(rolling_throughput)
                immediate_throughput_data.append(1 / step_time)
                # plot network usage
                cumulative_shard_download = np.sum(np.array(gen_step_downloads))
                network_data.append(cumulative_shard_download)
                steps.append(step + 1)

            # update plots and percentages at regular intervals
            plot_interval = math.ceil(total_batches / 15)
            if step == 1 or step % plot_interval == 0 or step == total_batches - 1:
                rolling_throughput_df = pd.DataFrame({
                    'step': steps,
                    'measurement': [' rolling avg'] * len(rolling_throughput_data),
                    'throughput (batches/s)': rolling_throughput_data
                })
                throughput_df = rolling_throughput_df
                network_df = pd.DataFrame({
                    'step': steps,
                    'cumulative network usage (bytes)': network_data
                })
                throughput_plot.altair_chart(get_line_chart(throughput_df, throughput_window,
                                                            True),
                                             use_container_width=True)
                network_plot.altair_chart(get_line_chart(network_df, throughput_window, False),
                                          use_container_width=True)
                # update progress bar and text
                percentage = int(100 * (step + 1) / (total_batches))
                status_text.text('%i%% Complete' % percentage)
                progress_bar.progress(percentage)

                # If applicable, check if the shuffle quality tasks are finished, and graph.
                if shuffle_quality and all(f.done() for f in futures) \
                    and not shuffle_quality_graphed:
                    display_shuffle_quality_graph(futures, shuffle_quality_plot)
                    shuffle_quality_graphed = True

        gen_step_times = np.array(gen_step_times)
        gen_step_downloads = np.array(gen_step_downloads)
        device_batch_size = dataset.get_batch_size()
        display_simulation_stats(sim_stats, total_batches, gen_step_times, time_per_sample,
                                 device_batch_size, time_to_first_batch, min_cache_limit,
                                 cache_limit)

        # If shuffle quality still hasn't been graphed yet, we get the result and graph it.
        if shuffle_quality and not shuffle_quality_graphed:
            display_shuffle_quality_graph(futures, shuffle_quality_plot)
            shuffle_quality_graphed = True


def get_input_params_initial(physical_nodes: int, devices: int, workers: int,
                             global_batch_size: int, train_dataset: dict, max_duration: Time,
                             time_per_sample: float, node_internet_bandwidth: Union[int, str]):
    """Create input_params and dataset for the first time.

    This function is called when modify_params is clicked, or when the run is submitted for
    simulation when using a yaml file.

    Args:
        physical_nodes (int): Number of physical nodes.
        devices (int): Number of devices per node.
        workers (int): Number of workers.
        global_batch_size (int): Global batch size.
        train_dataset (dict): Train dataset parameters.
        max_duration (Time): Maximum duration of simulation.
        time_per_sample (float): Time for one device to process one sample.
        node_internet_bandwidth (Union[int,str]): Internet bandwidth per node.
    """
    try:
        st.session_state['creating_dataset'] = True
        dataset = create_simulation_dataset(physical_nodes, devices, workers, global_batch_size,
                                            train_dataset)
        st.session_state['orig_dataset'] = dataset
        input_params = {}
        # dataset input_params
        input_params['streams'] = dataset.get_stream_info()
        # training input_params
        input_params['max_duration'] = max_duration
        input_params['epoch_size'] = dataset.get_epoch_size()
        input_params['device_batch_size'] = dataset.get_batch_size()
        # hardware and network input_params
        input_params['physical_nodes'] = physical_nodes
        input_params['devices'] = devices
        input_params['time_per_sample'] = time_per_sample
        input_params['node_network_bandwidth'] = node_internet_bandwidth
        # streaming input_params
        input_params['workers'] = workers
        input_params['canonical_nodes'] = dataset.get_num_canonical_nodes()
        input_params['predownload'] = dataset.get_predownload()
        input_params['shuffle'] = dataset.get_shuffle()
        input_params['shuffle_algo'] = dataset.get_shuffle_algo()
        input_params['shuffle_block_size'] = dataset.get_shuffle_block_size()
        input_params['seed'] = dataset.get_shuffle_seed()
        input_params['cache_limit'] = dataset.get_cache_limit()
        input_params['sampling_method'] = dataset.get_sampling_method()
        input_params['sampling_granularity'] = dataset.get_sampling_granularity()
        input_params['batching_method'] = dataset.get_batching_method()
        # Save input_params and originally created dataset to session state.
        st.session_state['input_params'] = input_params
    except FileNotFoundError:
        st.error('Please wait until the dataset is loaded before changing toggle values too \
                 quickly. Doing so can cause issues with creating multiple datasets, since \
                 Streamlit reloads widgets every single time a toggle value changes.',
                 icon='🚨')


# Define parameter input area.

# Check if the user wants to submit a yaml file.
use_yaml = col1.toggle(':sparkles: **Use `yaml`** :sparkles:', value=True)

if use_yaml:
    uploaded_yaml = col1.file_uploader('Upload a yaml file', type=['yaml'])
    if uploaded_yaml is not None:
        string_yaml = StringIO(uploaded_yaml.getvalue().decode('utf-8')).read()
        dict_yaml = yaml.safe_load(string_yaml)
        total_devices, workers, max_duration, global_batch_size, train_dataset = \
        ingest_yaml(yaml_dict=dict_yaml)
        # Check which parameters we still need to ask for.
        col1.write('The parameters below were not found in your yaml file. Enter them here:')
        physical_nodes = col1.number_input(
            'number of physical nodes',
            step=1,
            value=total_devices // 8 if total_devices is not None else 1,
            help=
            'number of physical nodes for this run. a node typically consists of 8 devices (GPUs).'
        )
        physical_nodes = int(physical_nodes)
        # Using physical_nodes, calculate number of devices per node.
        if total_devices is None:
            devices = col1.number_input(
                'devices per node',
                step=1,
                value=8,
                help=
                'number of devices (GPUs) per node for this run. there are typically 8 devices per node.'
            )
        else:
            if total_devices % physical_nodes != 0:
                raise ValueError('The number of devices must be divisible by the number of nodes.')
            devices = total_devices // physical_nodes
        devices = int(devices)
        time_per_sample = col1.number_input(
            'process time per sample (s)',
            step=0.0005,
            value=0.0175,
            format='%.4f',
            help='time for one device to process one sample from your dataset.')
        time_per_sample = float(time_per_sample)
        node_internet_bandwidth = col1.text_input('network bandwidth per node (bytes/s)',
                                                  value='500MB',
                                                  help='network bandwidth available to each \
                                                node. in practice, network bandwidth is \
                                                variable and is affected by many factors, \
                                                including cluster demand.')
        col1.warning(
            'If you are using a subclass of StreamingDataset with custom defaults, please \
                   select "Modify Parameters" and verify that the input parameters are correct.',
            icon='⚠️')
        submitted = col1.button('Simulate Run', use_container_width=True)
        shuffle_quality = col1.toggle('Analyze Shuffle Quality',
                                      value=False,
                                      help='Analyze shuffle qualities for this run for different \
                                        shuffle algos using an entropy-based metric. ⚠️ **Results \
                                        are *noisy estimates* and may not reflect the true \
                                        shuffle quality.**')
        modify_params = col1.toggle('Modify Parameters', value=False)

        # Display components and take actions based on the values of the above three buttons.
        if modify_params:
            # Create dataset and input_params if it doesn't already exist.
            if 'input_params' not in st.session_state:
                col1.write('Preparing dataset for modification...')
                get_input_params_initial(physical_nodes, devices, workers, global_batch_size,
                                         train_dataset, max_duration, time_per_sample,
                                         node_internet_bandwidth)
            # We have input_params in the session state. Use it to populate the form.
            defaults = st.session_state['input_params']
            # Define parameter input area with default values.
            input_params = {}
            param_inputs(col1, input_params, defaults=defaults)
            # input_params has been repopulated with new values. Save to session state.
            st.session_state['input_params'] = input_params

        if submitted:
            # Create dataset if it is not yet present.
            if 'input_params' not in st.session_state:
                col1.write('Preparing dataset for this run...')
                get_input_params_initial(physical_nodes, devices, workers, global_batch_size,
                                         train_dataset, max_duration, time_per_sample,
                                         node_internet_bandwidth)
            # If modify_params is false, we submit the jobs using the original dataset from yaml.
            if not modify_params:
                col1.write('Starting Simulation...')
                dataset = st.session_state['orig_dataset']
                # shuffle_quality is passed through to the job submission function.
                submit_jobs(shuffle_quality, dataset, time_per_sample, node_internet_bandwidth,
                            max_duration)
            else:
                # If modify_params is true, we retrieve the most recent input params from session
                # state, create a new dataset, and submit the jobs.
                col1.write('Preparing dataset with modifications...')
                # Get parameters for new SimulationDataset from input_params and train_dataset.
                input_params = st.session_state['input_params']
                train_dataset = get_train_dataset_params(input_params, old_params=train_dataset)
                # Get the rest of the needed params from the new inputs
                physical_nodes = input_params['physical_nodes']
                devices = input_params['devices']
                global_batch_size = input_params['device_batch_size'] * devices * physical_nodes
                workers = input_params['workers']
                max_duration = input_params['max_duration']
                time_per_sample = input_params['time_per_sample']
                node_internet_bandwidth = input_params['node_network_bandwidth']
                # Make sure node_internet_bandwidth is an int.
                dataset = create_simulation_dataset(physical_nodes, devices, workers,
                                                    global_batch_size, train_dataset)
                col1.write('Starting Simulation...')
                submit_jobs(shuffle_quality, dataset, time_per_sample, node_internet_bandwidth,
                            max_duration)
    else:
        # In this case, no file is uploaded, and we should clear dataset and input params if needed
        if 'input_params' in st.session_state:
            del st.session_state['input_params']
        if 'orig_dataset' in st.session_state:
            del st.session_state['orig_dataset']
else:
    submitted = col1.button('Simulate Run', use_container_width=True)
    col1.text('')
    shuffle_quality = col1.toggle('Analyze Shuffle Quality',
                                  value=False,
                                  help='Analyze shuffle qualities for this run for different \
                                        shuffle algos using an entropy-based metric. ⚠️ **Results \
                                        are *noisy estimates* and may not reflect the true \
                                        shuffle quality.**')
    if 'input_params' in st.session_state:
        st.session_state['input_params'] = {}
    input_params = {}
    param_inputs(col1, input_params, defaults=input_params)
    if submitted:
        # Params have been submitted. Create new dataset and proceed with simulation.
        col1.write('Preparing dataset for this run...')
        # Create index files and Stream object for each stream.
        streams = {}
        for stream_idx, stream in input_params['streams'].items():
            stream_dict = {}
            if 'path' in stream:
                # Case when user has provided a path to an index.json file.
                stream_folder = os.path.dirname(stream['path'])
                if stream['path_type'] == 'local':
                    stream_dict['local'] = stream_folder
                else:
                    stream_dict['remote'] = stream_folder
            else:
                # Case when user provides estimates for stream characteristics.
                index_path = create_stream_index(stream['shards'], stream['samples_per_shard'],
                                                 stream['avg_raw_shard_size'],
                                                 stream['avg_zip_shard_size'])
                stream_folder = os.path.dirname(index_path)
                stream_dict['local'] = stream_folder
            stream_dict['proportion'] = stream['proportion']
            stream_dict['repeat'] = stream['repeat']
            stream_dict['choose'] = stream['choose']
            streams[stream_idx] = stream_dict
        input_params['streams'] = streams
        # Get parameters for new SimulationDataset from input_params and train_dataset.
        train_dataset = get_train_dataset_params(input_params)
        # Get the rest of the needed params from the new inputs
        physical_nodes = input_params['physical_nodes']
        devices = input_params['devices']
        global_batch_size = input_params['device_batch_size'] * devices * physical_nodes
        workers = input_params['workers']
        max_duration = input_params['max_duration']
        time_per_sample = input_params['time_per_sample']
        node_internet_bandwidth = input_params['node_network_bandwidth']
        dataset = create_simulation_dataset(physical_nodes, devices, workers, global_batch_size,
                                            train_dataset)
        # Make sure input_params is in session state.
        st.session_state['input_params'] = input_params
        col1.write('Starting Simulation...')
        submit_jobs(shuffle_quality, dataset, time_per_sample, node_internet_bandwidth,
                    max_duration)
