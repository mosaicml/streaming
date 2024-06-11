# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Information about nodes, ranks, and workers."""

from typing import Any, Dict, Tuple

from torch.utils.data import get_worker_info
from typing_extensions import Self

from streaming.base import distributed as dist


class World:
    """Information about the nodes, ranks and workers of this run.

    .. warning::
      Be careful as to whether this object was initialized in a worker (if workers are used)
      or in a rank (which will claim one worker per rank).

    .. warning::
      In this World object, the counts (num_nodes, num_ranks, num_workers) are global -- not
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

    def __init__(
        self,
        num_nodes: int,
        ranks_per_node: int,
        workers_per_rank: int,
        worker: int,
    ) -> None:
        self.node = worker // (ranks_per_node * workers_per_rank)
        self.num_nodes = num_nodes
        self.is_multinode = 1 < num_nodes

        self.rank = worker // workers_per_rank
        self.num_ranks = num_nodes * ranks_per_node
        self.rank_of_node = self.rank % ranks_per_node
        self.ranks_per_node = ranks_per_node

        self.worker = worker
        self.num_workers = num_nodes * ranks_per_node * workers_per_rank
        self.worker_of_node = self.worker % (ranks_per_node * workers_per_rank)
        self.workers_per_node = ranks_per_node * workers_per_rank
        self.worker_of_rank = self.worker % workers_per_rank
        self.workers_per_rank = workers_per_rank
        self.is_leader = not worker
        self.is_local_leader = not self.worker_of_node

    def to_json(self) -> Dict[str, Any]:
        """Get a JSON version of this config.

        Returns:
            Dict[str, Any]: JSON config.
        """
        return dict(self.__dict__)

    @classmethod
    def _get_worker_info(cls) -> Tuple[int, int]:
        """Get worker info, or default to 0 of 1.

        Returns:
            Tuple[int, int]: Worker ID out of how many workers.
        """
        info = get_worker_info()
        if info:
            ret = info.id, info.num_workers
        else:
            ret = 0, 1
        return ret

    @classmethod
    def detect(cls) -> Self:
        """Detect the world state.

        Returns:
            Self: A new World state object according to dist and get_worker_info().
        """
        rank = dist.get_rank()
        ranks_per_node = dist.get_local_world_size()
        num_nodes = dist.get_world_size() // ranks_per_node
        worker_of_rank, workers_per_rank = cls._get_worker_info()
        worker = rank * workers_per_rank + worker_of_rank
        return cls(num_nodes, ranks_per_node, workers_per_rank, worker)

    def copy(self) -> Self:
        """Get a copy of this world state.

        Returns:
            Self: A new copy with the same state.
        """
        return World(
            num_nodes=self.num_nodes,
            ranks_per_node=self.ranks_per_node,
            workers_per_rank=self.workers_per_rank,
            worker=self.worker,
        )

    def replicate(self, replication: int) -> Self:
        """Get a copy of this world state with the given replication factor.

        Args:
            replication (int): Replication factor -- how many consecutive devices that should see
                the same samples..

        Returns:
            Self: A new sample replication version of this World state object.
        """
        if replication <= 0:
            raise ValueError(f'Replication factor must be positive.')

        if self.num_ranks % replication:
            raise ValueError(f'World size must be divisible by your replication factor.')

        rank = self.rank // replication  # Evenly divide ranks.
        num_ranks = self.num_ranks // replication  # Floor divide our rank.
        worker = rank * self.workers_per_rank + self.worker_of_rank  # Derive worker.
        # Attempt to evenly divide ranks per node. If not possible, the World
        # object will just use one node.
        if num_ranks % self.ranks_per_node == 0:
            num_nodes = num_ranks // self.ranks_per_node
        else:
            num_nodes = 1
        ranks_per_node = num_ranks // num_nodes  # Evenly divide ranks per node.
        return World(
            num_nodes=num_nodes,
            ranks_per_node=ranks_per_node,
            workers_per_rank=self.workers_per_rank,
            worker=worker,
        )

    def detect_workers(self) -> Self:
        """Get a copy of this world state with the worker information newly detected.

        Returns:
            Self: A new workers-newly-detected version of this World state object.
        """
        worker_of_rank, workers_per_rank = self._get_worker_info()
        worker = self.rank * workers_per_rank + worker_of_rank
        return World(
            num_nodes=self.num_nodes,
            ranks_per_node=self.ranks_per_node,
            workers_per_rank=workers_per_rank,
            worker=worker,
        )
