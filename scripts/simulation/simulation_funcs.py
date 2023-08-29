# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Functions for simulating streaming and displaying results."""

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from io import BytesIO
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from simulation.last_used_ordered_set import LastUsedOrderedSet
from sortedcollections import OrderedSet
import time

from streaming.base.partition import get_partitions
from streaming.base.shuffle import get_shuffle
from streaming.base.util import bytes_to_int, number_abbrev_to_int


def simulate(shards: int,
             samples_per_shard: Union[int, str],
             avg_shard_size: Union[float, str],
             device_batch_size: int,
             time_per_sample: float,
             batches_per_epoch: Union[int, str],
             epochs: int,
             physical_nodes: int,
             devices: int,
             node_network_bandwidth: Union[float,str],
             workers: int,
             canonical_nodes: int,
             predownload: Union[int, str],
             cache_limit: Union[int, str, None] = None,
             shuffle_algo: Optional[str] = None,
             shuffle_block_size: Union[int, str] = 1 << 18,
             seed: int = 42,
             generator: bool = False) -> Union[Tuple[int, int], Tuple[NDArray, NDArray], np.float64]:
    """Simulates step time and downloads using streaming for the specified input parameters.

    Key Notes and Assumptions:

       * assume that batch time is solely made up of two things: batch processing time and batch shard download wait time
       * loop through workers round-robin style for batches and for downloads
       * assume each node has a separate network bandwidth
       * the batch has to wait until all nodes have downloaded the shards containing batch samples.
       * for shard eviction itself, use LRU shard eviction to take out the least recently used shard, per node.
       * if a shard is unavailable, we use the normal behavior and wait for the regular downloading process to get it.
       * if a shard is available in a node, we just use it.
       * each node maintains an ordered set of shards that are present in the node
       * least recently used shard is the one at the front. most recently used is at the end of the ordered set.
       * when a shard is accessed during the batch it is moved to the end of the ordered set
       * when a shard is merely downloaded but not accessed for training, it goes to the front of the ordered set
       * if cache_limit is set, we check for going above cache limit at every download -- all downloads are assumed same size

    Args:
        shards (int): number of shards
        samples_per_shard (Union[int, str]): number of samples per shard
        avg_shard_size (Union[float, str]): average shard size (bytes)
        device_batch_size (int): device batch size (samples)
        time_per_sample (float): time to process one sample on one device (seconds)
        batches_per_epoch (Union[int, str]): number of batches per epoch
        epochs (int): number of epochs
        physical_nodes (int): number of physical nodes
        devices (int): number of devices per node
        node_network_bandwidth (Union[float, str]): network bandwidth per node (bytes/s)
        workers (int): number of workers per device
        canonical_nodes (int): number of canonical nodes
        predownload (Union[int, str]): number of samples to predownload per worker (samples)
        cache_limit (Union[int, str, None]): cache limit per node (bytes). Defaults to ``None``.
        shuffle_algo (str, optional): shuffling algorithm. Defaults to ``None``.
        shuffle_block_size (Union[int, str]): shuffling block size (samples). Defaults to ``1 << 18``.
        seed (int): shuffling seed. Defaults to ``42``.
        generator (bool): True if we yield throughput and shard_download one step at a time.

    Returns:
        step_times (NDArray): time taken by each step, calculated by simulation.
        shard_downloads (NDArray): amount of downloaded bytes at each step, calculated by simulation.
    """
    # simulation preparation...

    # tracking startup time
    start_time = time.time()
    startup_time = 0

    # make sure potential string args are usable
    samples_per_shard = number_abbrev_to_int(samples_per_shard)
    avg_shard_size = bytes_to_int(avg_shard_size)
    batches_per_epoch = number_abbrev_to_int(batches_per_epoch)
    node_network_bandwidth = bytes_to_int(node_network_bandwidth)
    predownload = number_abbrev_to_int(predownload)
    shuffle_block_size = number_abbrev_to_int(shuffle_block_size)
    if cache_limit:
        cache_limit = bytes_to_int(cache_limit)

    # we assume that each shard is going to be seen only once. Not handling up/down-sampling
    # or multiple streams for now.
    shard_sizes = np.array([samples_per_shard] * shards)

    # get partition of sample ids
    # structured as (physical nodes, ranks per node, workers per rank, batches per worker, batch size)
    orig_partitions = get_partitions(algo='orig',
                                num_samples=shards * samples_per_shard,
                                num_canonical_nodes=canonical_nodes,
                                num_physical_nodes=physical_nodes,
                                ranks_per_node=devices,
                                workers_per_rank=workers,
                                batch_size=device_batch_size,
                                drop_first=0)
    
    # time for the global batch is just device batch size * time per sample, since all devices process their microbatch in parallel
    avg_batch_time = device_batch_size * time_per_sample

    # simulate training!

    # loop over epochs, then batches...

    notification_batches = int(batches_per_epoch) / 20

    # track the shards which are present and evicted at each physical node
    node_shards = []
    node_evictions = []
    for _ in range(physical_nodes):
        node_shards.append(LastUsedOrderedSet())
        node_evictions.append(set())

    # node cache useages are initially nothin'
    node_cache_usage = np.array([0] * physical_nodes)

    # construct mapping of sample index -> shard number
    sample_to_shard = np.repeat(np.arange(shards), samples_per_shard)

    # track stats for each step
    step_times = []
    shard_downloads = []

    for epoch in range(epochs):

        if shuffle_algo is not None:
            # get shuffle of sample ids
            shuffle = get_shuffle(algo=shuffle_algo,
                                  shard_sizes=shard_sizes,
                                  num_canonical_nodes=canonical_nodes,
                                  seed=seed,
                                  epoch=epoch,
                                  block_size=shuffle_block_size)
            # index into the shuffle to get the new sample at each index
            partitions = np.where(orig_partitions != -1, shuffle[orig_partitions], -1)

        # handle initial predownload
        # reshape shuffled_partition to get samples, in order, per worker
        samples_per_worker = partitions.reshape(physical_nodes, devices, workers, -1)

        worker_sample_index = 0  # track which sample we are on. is an index per worker.
        worker_download_indices = np.array(
            [0] * physical_nodes
        )  # track which worker we are on for downloading, per node, round-robin style
        node_partial_shards = np.array([0] * physical_nodes).astype(
            np.float32)  # track partial shard downloads at each node

        # construct download shard OrderedSets for every worker
        # list of lists of OrderedSets. outer list is per node, inner list is per worker-device
        node_worker_downloads = []
        for physical_node in range(physical_nodes):
            worker_downloads = []
            # want to round-robin over devices, first, then workers so we don't only download samples from one device at a time
            for worker in range(workers):
                for device in range(devices):
                    download_samples = samples_per_worker[physical_node, device,
                                                          worker, :predownload]
                    # take out padded samples
                    download_samples = np.delete(download_samples,
                                                 np.where(download_samples == -1))
                    # get the shards these samples correspond to -- still want to maintain access order!
                    download_shards = OrderedSet(sample_to_shard[download_samples])
                    worker_downloads.append(download_shards)
            node_worker_downloads.append(worker_downloads)

        # if first epoch, add time so far to startup time
        if epoch == 0:
            startup_time += time.time() - start_time

        for batch_num in range(batches_per_epoch):

            if (batch_num + 1) % notification_batches == 0:
                print('Epoch: ' + str(epoch + 1) + ' | Batch ' + str(batch_num + 1) + '/' +
                      str(batches_per_epoch))

            # we round robin over workers per device. current worker is same across all nodes and devices
            curr_worker = batch_num % workers

            # track how long each node takes to download the shards that the current batch needs.
            node_batch_download_times = np.array([0] * physical_nodes)

            # track how many shards we downloaded in this batch total
            num_downloads = 0

            # get current samples and download samples for each node, for this batch
            for physical_node in range(physical_nodes):
                curr_batch_samples = samples_per_worker[physical_node, :, curr_worker,
                                                        worker_sample_index:worker_sample_index +
                                                        device_batch_size].flatten()

                #remove samples that are -1 (padded)
                curr_batch_samples = np.delete(curr_batch_samples,
                                               np.where(curr_batch_samples == -1))

                # get the shards these samples correspond to
                curr_batch_shards = set(sample_to_shard[curr_batch_samples])

                # shards we need to download is the set difference of shards already in node and current batch shards
                shards_needed = curr_batch_shards.difference(node_shards[physical_node].keys())

                # shards already present in the node -- we need to move these to the end of the node shards (we are using them).
                shards_present = curr_batch_shards.difference(shards_needed)

                # update all shards_present as accessed most recently in this node's shards
                for shard in shards_present:
                    # moves this shard to the end of the node shards
                    node_shards[physical_node].setuse(shard)

                # get the set of worker downloads for the current node
                worker_downloads = node_worker_downloads[physical_node]

                # push the download range for the current worker forward by device_batch_size
                for device in range(devices):
                    # only for current workers in batch, add any potential new shards to (pre)download.
                    # only the current workers have their (pre)download range moved at the current step.
                    new_download_samples = samples_per_worker[physical_node, device, curr_worker,
                                                              worker_sample_index +
                                                              predownload:worker_sample_index +
                                                              device_batch_size + predownload]

                    #remove samples that are -1 (padded)
                    new_download_samples = np.delete(new_download_samples,
                                                     np.where(new_download_samples == -1))

                    # get the shards these samples correspond to, maintaining access order
                    new_download_shards = OrderedSet(sample_to_shard[new_download_samples])

                    # get set of curr_worker downloads per device
                    worker_download = worker_downloads[curr_worker * devices + device]

                    # add these new shards to the predownload ONLY if they are not already in the node
                    # won't be any duplicates in the OrderedSet of worker downloads anyways.
                    for shard in new_download_shards:
                        if shard not in node_shards[physical_node]:
                            worker_download.add(shard)

                # get the current worker we are starting downloads from
                curr_worker_download_index = worker_download_indices[physical_node]

                # num_batch_shards is different from len(shards_needed) because there is no guarantee that the shards we need
                # are immediately downloaded first. Other shards from other workers may get downloaded before we download
                # the shards needed for the current batch.
                num_batch_shards = 0

                # if we need shards for the current batch, we loop through worker downloads until there are no more shards needed
                while len(shards_needed) > 0:
                    # traverse worker_downloads until we have a worker that has samples to predownload
                    empty_download_counter = 0
                    while len(worker_downloads[curr_worker_download_index]
                             ) == 0 and empty_download_counter < devices * workers:
                        empty_download_counter += 1
                        curr_worker_download_index = (curr_worker_download_index + 1) % (devices *
                                                                                         workers)

                    # break out of predownload loop if no workers in the node have any predownloads.
                    if empty_download_counter >= devices * workers:
                        break

                    # get the worker that has samples to predownload
                    worker_download = worker_downloads[curr_worker_download_index]

                    # first entry in predownload is the next shard the worker wants
                    download_shard = worker_download[0]

                    if download_shard not in node_shards[physical_node]:
                        # handle possible eviction
                        if cache_limit and node_cache_usage[
                                physical_node] + avg_shard_size > cache_limit:
                            # evict the LRU shard
                            node_shards[physical_node].popLRU()
                            # update the node cache usage
                            node_cache_usage[physical_node] -= avg_shard_size
                        num_batch_shards += 1
                        num_downloads += 1
                        node_cache_usage[physical_node] += avg_shard_size
                        # add this shard to node_shards for the node that the worker is on
                        # second param as False means we don't move the shard to the end of the OrderedDict (we haven't actually used the shard yet)
                        # but if the shard is in shards_needed, we did actually use the shard and so we move it to the end of the node shards.
                        node_shards[physical_node].setitem(download_shard, True)
                        # if shard must have been in shards_needed, remove it
                        shards_needed.discard(download_shard)
                        # if shard used to be in node eviction list, remove it -- it's now present
                        node_evictions[physical_node].discard(download_shard)

                    # discard from worker_download
                    worker_download.discard(download_shard)

                    # increment download index
                    curr_worker_download_index = (curr_worker_download_index + 1) % (devices *
                                                                                     workers)

                # calculate how much time we spent downloading shards for this node for the current batch only
                batch_download_time = (num_batch_shards * avg_shard_size) / node_network_bandwidth
                node_batch_download_times[physical_node] = batch_download_time

                # update worker download index for this node
                worker_download_indices[physical_node] = curr_worker_download_index

            # The batch will only start once all nodes have all the samples for the
            # batch ready. So the true start time of the batch is determined by the longest batch_download_time
            # over all nodes. And that means the download_time_left for nodes that finish earlier will be longer
            # and only the slowest node will have a download_time_left of avg_batch_time.
            slowest_download_time = np.max(node_batch_download_times)

            # if we are on the first step, add slowest_download_time to startup time
            if epoch == 0 and batch_num == 0:
                startup_time += slowest_download_time
            
            for physical_node in range(physical_nodes):

                # we will always have the avg_batch_time to do more downloads, plus whatever amount of time this node finished early
                download_time_left = avg_batch_time + (slowest_download_time -
                                                       node_batch_download_times[physical_node])

                # get number of bytes/shards/remainder we can download in predownload_time
                download_bytes_left = node_network_bandwidth * download_time_left

                # number of shards we can download right now --
                # add in the fractional part of shard that may have been downloading from previous step
                download_shards_left = (
                    (download_bytes_left) / avg_shard_size) + node_partial_shards[physical_node]

                # get the current worker we are starting downloads from
                curr_worker_download_index = worker_download_indices[physical_node]

                # get the set of worker downloads for the current node
                worker_downloads = node_worker_downloads[physical_node]

                # while we can still download a whole shard, we keep predownloading shards in the allotted time.
                while download_shards_left > 1:
                    # traverse worker_downloads until we have a worker that has samples to predownload
                    empty_download_counter = 0
                    while len(worker_downloads[curr_worker_download_index]
                             ) == 0 and empty_download_counter < devices * workers:
                        empty_download_counter += 1
                        curr_worker_download_index = (curr_worker_download_index + 1) % (devices *
                                                                                         workers)

                    # break out of predownload loop if no workers in the node have any predownloads.
                    if empty_download_counter >= devices * workers:
                        break

                    # get the worker that has samples to predownload
                    worker_download = worker_downloads[curr_worker_download_index]

                    # first entry in predownload is the next shard the worker wants
                    download_shard = worker_download[0]

                    if download_shard not in node_shards[physical_node]:
                        # handle possible eviction
                        if cache_limit and node_cache_usage[
                                physical_node] + avg_shard_size > cache_limit:
                            # evict the LRU shard
                            node_shards[physical_node].popLRU()
                            # update the node cache usage
                            node_cache_usage[physical_node] -= avg_shard_size
                        num_downloads += 1
                        node_cache_usage[physical_node] += avg_shard_size
                        # add this shard to node_shards for the node that the worker is on
                        # second param is False because this shard wasn't actually needed for the current batch. doesn't count as an actual access.
                        node_shards[physical_node].setitem(download_shard, False)
                        # decrement download_shards_left because we actually downloaded something
                        download_shards_left -= 1

                    # discard from worker_download
                    worker_download.discard(download_shard)

                    # increment download index
                    curr_worker_download_index = (curr_worker_download_index + 1) % (devices *
                                                                                     workers)

                # insert download_shards_left into node_partial_shards
                node_partial_shards[physical_node] = download_shards_left

                # update worker download index for this node
                worker_download_indices[physical_node] = curr_worker_download_index

            if generator:
                yield (slowest_download_time + avg_batch_time, avg_shard_size*num_downloads)
            else:
                step_times.append(slowest_download_time + avg_batch_time)
                shard_downloads.append(avg_shard_size*num_downloads)

            # if we are at last worker, then the sample_index per worker should shift ahead by device_batch_size
            if curr_worker == workers - 1:
                worker_sample_index += device_batch_size
    
    if not generator:
        step_times = np.array(step_times)
        shard_downloads = np.array(shard_downloads)
        yield step_times, shard_downloads, startup_time
    else:
        yield startup_time

def get_rolling_avg_throughput(step_times: NDArray, window: int = 10) -> NDArray:
    step_times_rolling_avg = np.convolve(step_times, np.ones(window) / window, mode='valid')
    batch_throughput_rolling_avg = 1 / step_times_rolling_avg
    batch_throughput_rolling_avg = np.concatenate((np.array([0] * (window-1)), batch_throughput_rolling_avg))

    return batch_throughput_rolling_avg


def plot_simulation(step_times: NDArray,
                    shard_downloads: NDArray,
                    web: bool = True,
                    window: int = 10) -> Optional[bytes]:
    """Plots simulation results for web UI or local script.

    Args:
        step_times (NDArray): time per step, as calculated by simulation
        shard_downloads (NDArray): download size (bytes) per step, as calculated by simulation
        web (bool, optional): True if displaying on web UI, False if displaying through local script. Defaults to `True``.
        window (int, optional): window size to calculate batch throughput over. Defaults to ``10``.

    Returns:
        Optional[bytes]: bytes of plot image if ``web`` is ``True``, else plot is displayed, and returns ``None``.
    """
    import matplotlib
    if web:
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    immediate_batch_throughput = 1 / step_times

    shard_downloads_cumulative = np.cumsum(shard_downloads)

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

    ax2.plot(np.arange(shard_downloads_cumulative.shape[0]),
             shard_downloads_cumulative,
             color='blue',
             label='total')
    ax2.set_ylim([0, max(shard_downloads_cumulative) * 1.1])
    ax2.set_xlabel('step')
    ax2.set_ylabel('cumulative download (bytes)')
    ax2.set_title('network traffic (bytes)')

    fig.set_figheight(8)
    fig.set_figwidth(6)

    if web:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=fig.dpi)
        buf.seek(0)
        return buf.read()
    else:
        plt.show()
        return None
    

def get_simulation_stats(step_times, shard_downloads, time_per_sample, device_batch_size):
    """Gets simulation stats for web UI.

    Args:
        step_times (NDArray): time per step, as calculated by simulation
        shard_downloads (NDArray): download size (bytes) per step, as calculated by simulation

    Returns:
        Tuple[float, float, float]: percent of download-limited steps, warmup time
    """
    
    # calculate percent of download-limited steps
    min_step_time = time_per_sample * device_batch_size
    all_throughput_drops = np.count_nonzero(step_times > (min_step_time))

    # calculate warmup time (time to first max possible rolling average throughput)
    max_throughput = 1 / min_step_time
    rolling_avg_throughput = get_rolling_avg_throughput(step_times)
    if np.max(rolling_avg_throughput) == max_throughput:
        warmup_step = np.argmax(rolling_avg_throughput >= (max_throughput)) + 1
        warmup_time = np.sum(step_times[:warmup_step])
    else:
        # we never hit the max possible throughput
        warmup_step = rolling_avg_throughput.shape[0]
        warmup_time = np.sum(step_times)
    
    # see if there are throughput drops after warmup so we can notify users
    if warmup_step != rolling_avg_throughput.shape[0]:
        # if we did hit the max throughput then we check for later drops
        post_warmup_throughput_drops = np.count_nonzero(step_times[warmup_step:] > min_step_time)
    else:
        # since warmup was the whole time, there are no post-warmup throughput drops
        post_warmup_throughput_drops = 0
    
    return all_throughput_drops, warmup_time, warmup_step, post_warmup_throughput_drops


    
