# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Functions for simulating shard downloads and calculating needed cache limit for downloads."""

from typing import Optional, Tuple

import numpy as np
from core.node_tracker import NodeTracker
from numpy.typing import NDArray


def simulate_shard_downloads(node: NodeTracker,
                             raw_shard_sizes: NDArray[np.int64],
                             zip_shard_sizes: NDArray[np.int64],
                             current_batch_downloads: bool,
                             step_num: int,
                             cache_limit: Optional[int] = None,
                             shards_needed: Optional[set] = None,
                             download_bytes_left: Optional[int] = None) -> Tuple[str, int]:
    """Simulate downloading a shard for a node.

    Args:
        node (NodeTracker): The node to simulate downloading a shard for.
        raw_shard_sizes (NDArray[np.int64]): The raw sizes of all shards.
        zip_shard_sizes (NDArray[np.int64]): The zip sizes of all shards.
        current_batch_downloads (bool): Whether we are downloading shards for the current batch.
        step_num (int): The current step number.
        cache_limit (Optional[int]): The cache limit for the node. Defaults to ``None``.
        shards_needed (Optional[set]): The shards needed for the current batch.
            Defaults to ``None``.
        download_bytes_left (Optional[int]): The number of download bytes left in the downloading
            time interval. Defaults to ``None``.

    Returns:
        Tuple[bool, int]: A tuple of the shard download status and the download size.
    """
    worker_download = node.get_next_worker_with_downloads()
    if worker_download is None:
        # No workers have shards to download.
        return ('empty', 0)

    # Proceed with downloading the shard for this worker.
    download_shard = int(worker_download[0])

    # Get the raw and zip sizes, in bytes, of the shard.
    shard_raw_size = int(raw_shard_sizes[download_shard])
    shard_zip_size = int(zip_shard_sizes[download_shard])
    # If shard is compressed, we download the zipped size. Otherwise, download raw size.
    download_size = shard_zip_size or shard_raw_size

    # If we are not downloading for the current batch, we need to check if the download bytes
    # left is sufficient to download this shard. Otherwise, we have to keep downloading anyways.
    bytes_sufficient = True
    if not current_batch_downloads:
        if download_bytes_left is not None:
            bytes_sufficient = (download_size <= download_bytes_left)
        else:
            raise ValueError('Must specify download_bytes_left if not downloading for \
                             current batch.')

    if download_shard not in node.shards and bytes_sufficient:
        # Shard is not present in node, so we download it.
        # Handle possible shard eviction.
        if cache_limit and node.cache_usage + shard_raw_size > cache_limit:
            # Evict shards until we have space for this shard.
            node.evict_until_satisfied(shard_raw_size, raw_shard_sizes)

        # Shards are assumed to be decompressed on download, so cache_usage increases by raw size.
        node.cache_usage += shard_raw_size

        # If we are downloading shards for the current batch, we need to check if the shard
        # is needed by the current batch. If it is, we make sure to mark it as most recently used.
        # If we are not downloading shards for the current batch then no shard is marked as used.
        if current_batch_downloads:
            if shards_needed is not None:
                node.add_shard(download_shard)
                shards_needed.discard(download_shard)
            else:
                raise ValueError('Must specify shards_needed if downloading for current batch.')
        else:
            node.add_shard(download_shard)

        if node.shard_access_starts[download_shard] == -1:
            # Shard has never been accessed before. Set its access start.
            node.shard_access_starts[download_shard] = step_num
        # For any shard access, we are accessing the shard so we need the shard until
        # at least the next step begins. Adding 0.5 ensures that we evict shards
        # after they are used for the last time, but before they are replaced by
        # new downloads in the next step.
        node.shard_access_ends[download_shard] = step_num + 0.5

        # Advance the worker download index.
        node.increment_worker_download_index()
        # We have now downloaded this shard. Remove from worker download queue.
        worker_download.pop()
        return ('downloaded', download_size)
    elif not (current_batch_downloads) and download_shard not in node.shards \
        and download_bytes_left is not None:
        # This is the case when we are not downloading for the current batch, and need to download
        # a shard but do not have the download bytes to fully download the shard this step.
        # We do not advance the worker download index since we still are downloading this shard.
        node.partial_shard_bytes = download_bytes_left
        # We only account for downloaded bytes when we fully download a shard.
        return ('partial', 0)
    else:
        # The shard is already present in the node. Advance the worker download index.
        node.increment_worker_download_index()
        # Node already has this shard. Remove from worker download queue.
        worker_download.pop()
        return ('present', 0)


def run_cache_limit(nodes: list[NodeTracker], raw_shard_sizes: NDArray[np.int64]) -> int:
    """Find the minimum needed cache limit across all nodes for this run.

    Args:
        nodes (list[NodeTracker]): The nodes, which contain shard use information.
        raw_shard_sizes (NDArray[np.int64]): The raw sizes of all shards.

    Returns:
        int: The minimum needed cache limit, in bytes, for the run.
    """
    # Find the overall needed cache usage, as the max needed for any node at any point.
    needed_cache_usage = 0
    for node in nodes:
        # For each node, create its own list of shard access events.
        # Access event tuples are (event time, shard idx, event type)
        # Event types: 0 means a shard has been accessed, 1 means a shard has ended access.
        node_shards = node.get_all_shards()
        access_events = []
        access_events += [(node.shard_access_starts[i], i, 0) for i in node_shards]
        access_events += [(node.shard_access_ends[i], i, 1) for i in node_shards]

        # Sort the access events to get shard events, in order.
        access_events.sort(key=lambda x: x[0])

        # For each shard event, update the cache usage. Assume that shards are decompressed
        # immediately on download, so use raw shard sizes.
        curr_cache_usage = 0
        for event in access_events:
            if event[2] == 0:
                # Shard has been accessed. Increment cache usage.
                curr_cache_usage += raw_shard_sizes[event[1]]
                # Needed cache usage is the max of all cache usages across the run.
                if curr_cache_usage > needed_cache_usage:
                    needed_cache_usage = curr_cache_usage
            else:
                # Shard access has ended. Decrement cache usage.
                curr_cache_usage -= raw_shard_sizes[event[1]]

    return needed_cache_usage
