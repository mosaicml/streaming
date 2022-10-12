# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Information about nodes, ranks, and workers."""

from torch.utils.data import get_worker_info

from streaming.base.distributed import get_local_world_size, get_rank, get_world_size
from streaming.base.interprocess import Barrier


class World:
    """Information about nodes, ranks and workers.

    Warning: be careful as to whether this object was initialized in a worker (if workers are used)
    or in a rank (which will claim one worker per rank).
    """

    def __init__(self):
        self.rank = get_rank()
        self.num_ranks = get_world_size()
        self.ranks_per_node = get_local_world_size()
        self.rank_of_node = self.rank % self.ranks_per_node
        self.node = self.rank // self.ranks_per_node
        self.num_nodes = self.num_ranks // self.ranks_per_node

        info = get_worker_info()
        if info:
            self.worker_of_rank = info.id
            self.workers_per_rank = info.num_workers
        else:
            self.worker_of_rank = 0
            self.workers_per_rank = 1

        self.worker_of_node = self.rank_of_node * self.workers_per_rank + self.worker_of_rank
        self.workers_per_node = self.ranks_per_node * self.workers_per_rank
        self.worker = self.rank * self.workers_per_rank + self.worker_of_rank
        self.num_workers = self.num_ranks * self.workers_per_rank

        self.is_multinode = 1 < self.num_nodes
        self.is_local_leader = not self.worker_of_node
        self.is_leader = not self.worker

        self.barrier = Barrier(self.workers_per_node, 'barrier')
