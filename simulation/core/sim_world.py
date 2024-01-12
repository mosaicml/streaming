# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Contains info about the nodes, ranks, and workers of the run for simulation purposes."""

from streaming.base.world import World


class SimulationWorld(World):
    """Contains info about the nodes, ranks, and workers of the run, for simulation.

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
    Args:
        nodes (int): The number of nodes.
        devices (int): The number of devices per node.
        workers (int): The number of workers per device.
    """

    def __init__(self, nodes: int, devices: int, workers: int):

        # For simulation purposes, we take in the nodes, devices, and workers from the
        # SimulationDataset, and assume we are always rank 0 and worker 0.
        self.rank = 0
        self.num_ranks = nodes * devices
        self.ranks_per_node = devices
        self.rank_of_node = self.rank % self.ranks_per_node
        self.node = self.rank // self.ranks_per_node
        self.num_nodes = self.num_ranks // self.ranks_per_node
        self.is_multinode = 1 < self.num_nodes

        self.worker_of_rank = 0
        self.workers_per_rank = workers

        self.worker = self.rank * self.workers_per_rank + self.worker_of_rank
        self.num_workers = self.num_ranks * self.workers_per_rank
        self.worker_of_node = self.rank_of_node * self.workers_per_rank + self.worker_of_rank
        self.workers_per_node = self.ranks_per_node * self.workers_per_rank

        self.is_leader = not self.worker
        self.is_local_leader = not self.worker_of_node
