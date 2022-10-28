# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Information about nodes, ranks, and workers."""

from torch.utils.data import get_worker_info

from streaming.base import distributed as dist


class World:
    """Information about the nodes, ranks and workers of this run.

    Warning: Be careful as to whether this object was initialized in a worker (if workers are used)
    or in a rank (which will claim one worker per rank).

    Warning: In this World object, the counts (num_nodes, num_ranks, num_workers) are global -- not
    to be confused with DataLoader num_workers, which is per rank.

    Nodes are all assumed to contain the same number of devices (via local_world_size).

    Nodes:
      - node / num_nodes
      - is_multinode

    Ranks:
      - rank / num_ranks
      - rank_of_node / ranks_per_node

    Workers:
      - worker / num_workers
      - worker_of_node / workers_per_node
      - worker_of_rank / workers_per_rank
      - is_leader
      - is_local_leader
    """

    def __init__(self):
        self.rank = dist.get_rank()
        self.num_ranks = dist.get_world_size()
        self.ranks_per_node = dist.get_local_world_size()
        self.rank_of_node = self.rank % self.ranks_per_node

        self.node = self.rank // self.ranks_per_node
        self.num_nodes = self.num_ranks // self.ranks_per_node
        self.is_multinode = 1 < self.num_nodes

        info = get_worker_info()
        if info:
            self.worker_of_rank = info.id
            self.workers_per_rank = info.num_workers
        else:
            self.worker_of_rank = 0
            self.workers_per_rank = 1

        self.worker = self.rank * self.workers_per_rank + self.worker_of_rank
        self.num_workers = self.num_ranks * self.workers_per_rank
        self.worker_of_node = self.rank_of_node * self.workers_per_rank + self.worker_of_rank
        self.workers_per_node = self.ranks_per_node * self.workers_per_rank

        self.is_leader = not self.worker
        self.is_local_leader = not self.worker_of_node
