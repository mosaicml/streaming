# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples to nodes/ranks/workers for elastically deterministic sample order."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from streaming.base.batching.device_per_stream import generate_work_device_per_stream_batching
from streaming.base.batching.per_stream import generate_work_per_stream_batching
from streaming.base.batching.random import generate_work_random_batching
from streaming.base.batching.stratified import generate_work_stratified_batching
from streaming.base.world import World

if TYPE_CHECKING:
    from streaming.base.dataset import StreamingDataset

batching_methods = {
    'random': generate_work_random_batching,
    'stratified': generate_work_stratified_batching,
    'per_stream': generate_work_per_stream_batching,
    'device_per_stream': generate_work_device_per_stream_batching,
}


def generate_work(batching_method: str, dataset: StreamingDataset, world: World, epoch: int,
                  sample_in_epoch: int) -> NDArray[np.int64]:
    """Apportion shards/samples to nodes/ranks/workers for elastically deterministic sample order.

    Args:
        batching_method (str): The batching method to use.
        dataset (StreamingDataset): Dataset to generate the partition for.
        world (World): World state.
        epoch (int): Which epoch it is.
        sample_in_epoch (int): Where we are in the epoch.

    Returns:
        NDArray[np.int64]: The epoch (num physical nodes, ranks per node, workers per rank,
            batches per worker, batch size).
    """
    get = batching_methods[batching_method]
    return get(dataset, world, epoch, sample_in_epoch)
