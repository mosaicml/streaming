# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Functions for simulating streaming and displaying results."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from numpy.typing import NDArray
from core.shard_downloads import simulate_shard_downloads, run_cache_limit
from core.sim_time import Time
from typing import Tuple, Union
import time

from core.simulation_dataset import SimulationDataset
from core.node_tracker import NodeTracker
from core.utils import get_batches_epochs, bytes_to_time, time_to_bytes

def simulate(dataset: SimulationDataset,
             time_per_sample: float,
             node_network_bandwidth: Union[float,str],
             generator: bool = False,
             max_duration: Time = None
             ) -> Union[Tuple[int, float, int], 
                        Tuple[NDArray, NDArray, float, int], 
                        Tuple[float, int]]:
    """Simulates step time and downloads using streaming for the specified input parameters.

    Key Notes and Assumptions:

       * assume that batch time is solely made up of two things: batch processing time and batch
         shard download wait time
       * loop through workers round-robin style for batches and for downloads
       * assume each node has a separate network bandwidth
       * the batch has to wait until all nodes have downloaded the shards containing batch samples.
       * for shard eviction itself, use LRU shard eviction to take out the least recently used
         shard, per node.
       * shards are shared across devices on a single node, but nodes do not share shards between
         each other.
       * if a shard is unavailable, we wait for some worker to download it.
       * if a shard is available in a node, we just use it.

    Args:
        dataset (SimulationDataset): SimulationDataset object created based on input params/yaml
        time_per_sample (float): time to process one sample on one device (seconds)
        node_network_bandwidth (Union[float, str]): network bandwidth per node (bytes/s)
        generator (bool): True if we yield throughput and shard_download one step at a time.
        max_duration (Time, optional): max duration of simulation. Defaults to ``None``.

    Returns:
        Union[Tuple[int, int], Tuple[NDArray, NDArray], np.float64]: either a Tuple of step_time,
            shard_download, or a Tuple all step_times, shard_downloads or the startup time.
    """

    # tracking startup time, which includes SimulationDataset instantiation time.
    start_time = time.time()
    startup_time = dataset.get_instantiation_time()

    # Get batches, epochs, total batches from dataset and provided time info.
    batches_per_epoch, epochs, total_batches = get_batches_epochs(dataset, max_duration)
        
    # Retrieve streaming and dataset information from SimulationDataset.
    physical_nodes = dataset.get_nodes()
    devices = dataset.get_devices()
    workers = dataset.get_workers()
    device_batch_size = dataset.get_batch_size()
    predownload = dataset.get_predownload()
    cache_limit = dataset.get_cache_limit()
    total_shards = dataset.get_num_shards()
    raw_shard_sizes = dataset.get_raw_shard_sizes()
    zip_shard_sizes = dataset.get_zip_shard_sizes()
    # dataset's spanner object maps global sample id to shard id.
    sample_to_shard = dataset.get_spanner()

    # track shard access ranges to compute minimum needed cache limit for the run.
    shard_access_starts = np.full(total_shards, -1)
    shard_access_ends = np.full(total_shards, -1)

    # Initialize NodeTracker objects for each node. These keep track of shards, worker downloads,
    # cache usage, etc. for each node.
    nodes = []
    for _ in range(physical_nodes):
        nodes.append(NodeTracker(workers, devices, predownload, device_batch_size, cache_limit))
    
    # Time for the global batch is just device batch size * time per sample.
    # We assume all devices process their microbatch perfectly in parallel.
    avg_batch_process_time = device_batch_size * time_per_sample

    notification_batches = int(batches_per_epoch) / 20

    # Simulate training by looping over epochs and batches.

    # Track time and downloads at each step.
    step_times = []
    step_downloads = []

    for epoch in range(epochs):
        
        # Get the samples, divided up per node, for this epoch.
        samples_per_node = dataset.get_samples_per_node(epoch, 0)

        # Set the samples for each node for this epoch.
        for node_id, node in enumerate(nodes):
            node.samples = samples_per_node[node_id]
            node.initialize_worker_downloads(sample_to_shard)

        # Track which sample we are currently on, as a worker id. We round-robin over workers.
        worker_sample_index = 0
        
        # if first epoch, add time so far to startup time
        if epoch == 0:
            startup_time += time.time() - start_time
        
        # Iterate over batches
        for batch in range(batches_per_epoch):

            step_num = batch + (batches_per_epoch * epoch)

            # If we are at the last batch, exit.
            if step_num >= total_batches:
                break

            # Print progress every notification_batches interval.
            if (batch + 1) % notification_batches == 0:
                print('Epoch: ' + str(epoch + 1) + ' | Batch ' + str(batch + 1) + '/' +
                      str(batches_per_epoch))
            
            # We round-robin over workers per device. The current batch's worker is the same
            # across every device.
            curr_worker = batch % workers
            # Track how long each node takes to download the shards that the current batch needs.
            node_batch_download_times = np.array([0] * physical_nodes)
            # Track how many total bytes are downloaded in this batch by all nodes.
            downloaded_bytes = 0

            # Get current samples and download samples for each node, for this batch.
            for node_id, node in enumerate(nodes):

                shards_needed, shards_present = node.get_current_batch_shards(curr_worker, 
                                                                              worker_sample_index, 
                                                                              sample_to_shard)
                # Mark all shards present as accessed most recently in this node.
                node.set_shards_used(shards_present, shard_access_ends, step_num)
                # Push the predownload for the current batch workers ahead by device_batch_size.
                node.update_worker_predownloads(curr_worker, worker_sample_index, sample_to_shard)
                # Track bytes downloaded by this node.
                node_downloaded_bytes = 0

                # Because we assume downloads also round-robin over workers, we can download shards
                # other than the current batch's shards while looking for current batch's shards.
                while len(shards_needed) > 0:
                    download_outcome, download_size = \
                        simulate_shard_downloads(node, 
                                                 raw_shard_sizes, 
                                                 zip_shard_sizes, 
                                                 current_batch_downloads=True,
                                                 shard_access_starts=shard_access_starts,
                                                 shard_access_ends=shard_access_ends,
                                                 step_num=step_num,
                                                 cache_limit=cache_limit,
                                                 shards_needed=shards_needed)
                    if download_outcome == "downloaded":
                        node_downloaded_bytes += download_size
                        downloaded_bytes += download_size
                    elif download_outcome == "present":
                        # If the shard was already present in the node, continue downloading.
                        pass
                    else:
                        # If no shard downloads are left in the node, stop downloading.
                        break
                                
                # Calculate how much time this node spent downloading shards
                node_batch_download_times[node_id] = bytes_to_time(node_downloaded_bytes,
                                                                   node_network_bandwidth)
            
            # The node that took the longest to download shards is the bottleneck. All other nodes
            # use the extra time to continue downloading.
            slowest_download_time = np.max(node_batch_download_times)

            # If we are on the first step, add slowest_download_time to startup time
            if epoch == 0 and batch == 0:
                startup_time += slowest_download_time

            # Iterate over nodes again to continue downloading shards.
            for node_id, node in enumerate(nodes):
                # The download time each node has is the avg_batch_process_time plus the extra time
                # the node has from finishing earlier than the slowest node.
                download_time_left = avg_batch_process_time + (slowest_download_time -
                                                                node_batch_download_times[node_id])
                # Get number of bytes we can download in download_time_left.
                # We also include any partially downloaded bytes from previous steps.
                download_bytes_left = time_to_bytes(download_time_left, node_network_bandwidth) + \
                    node.partial_shard_bytes

                while True:
                    download_outcome, download_size = \
                        simulate_shard_downloads(node, 
                                                 raw_shard_sizes, 
                                                 zip_shard_sizes, 
                                                 current_batch_downloads=False,
                                                 shard_access_starts=shard_access_starts,
                                                 shard_access_ends=shard_access_ends,
                                                 step_num=step_num,
                                                 cache_limit=cache_limit,
                                                 download_bytes_left=download_bytes_left)
                    if download_outcome == "downloaded":
                        downloaded_bytes += download_size
                        download_bytes_left -= download_size
                    elif download_outcome == "present":
                        pass
                    else:
                        # If no shard downloads are left in the node, or we could only partially
                        # download a shard, stop downloading.
                        break
            
            # Yield or store step number, time and download for this step
            if generator:
                yield step_num, slowest_download_time + avg_batch_process_time, downloaded_bytes
            else:
                step_times.append(slowest_download_time + avg_batch_process_time)
                step_downloads.append(downloaded_bytes)
            
            # Increment worker_sample_index by device_batch_size if we are at the last worker.
            # As we round-robin over workers, the sample index per worker is increased as we loop
            # through all workers.
            if curr_worker == workers - 1:
                worker_sample_index += device_batch_size
    
    # Simulation is finished. Calculate needed cache limit from shard access ranges.
    min_cache_limit = run_cache_limit(shard_access_starts, shard_access_ends,
                                      raw_shard_sizes, nodes)

    # Yield results.
    if not generator:
        step_times = np.array(step_times)
        step_downloads = np.array(step_downloads)
        yield step_times, step_downloads, startup_time, min_cache_limit
    else:
        yield startup_time, min_cache_limit