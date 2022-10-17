# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A non-streaming pytorch IterableDataset with mid-epoch resumption."""

# TODO: checkpointing needs to save/restore the epoch of each of the other Datasets too
# TODO: when resuming, trainer progressbar should reflect our place in shortened epoch

import json
import os
from multiprocessing.shared_memory import SharedMemory
from time import sleep
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import distributed as dist
from torch.utils.data import IterableDataset

from streaming.base.format import reader_from_json
from streaming.base.index import Index
from streaming.base.shuffle import get_epoch
from streaming.base.world import World


class LocalResumableDataset(IterableDataset):
    """A resumable streaming dataset whose shards reside locally as a pytorch IterableDataset.

    Args:
        local (str): Local dataset directory where the dataset is present.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to shuffle the samples while iterating. Defaults to ``True``.
        seed (int): Base random seed, used for shuffling.
    """

    def __init__(self,
                 local: str,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 seed: int = 42):
        self.local = local
        self.split = split or ''
        self.shuffle = shuffle
        self.seed = seed

        filename = os.path.join(local, split, 'index.json')  # pyright: ignore
        obj = json.load(open(filename))
        assert obj['version'] == 2

        self.shards = []
        for info in obj['shards']:
            shard = reader_from_json(local, split, info)
            self.shards.append(shard)

        self.shard_sizes = np.array([x.samples for x in self.shards])
        self.index = Index(self.shard_sizes)

        # TODO: add a coordinated random integer prefix to all shm names for uniqueness

        '''
        world = World()
        device = torch.device(f'cuda:{world.rank_of_node}')
        tensor = torch.zeros(1, dtype=torch.int64, device=device)
        if world.is_leader:
            tensor[0] = np.random.randint(1 << 30)
        dist.broadcast(tensor, 0)
        self.shm_prefix = tensor.item()
        '''

        self.resume_shm = None

        shm_name = f'{self.split}_epoch'
        shm_bytes = np.int64().nbytes
        try:
            self._epoch_shm = SharedMemory(shm_name, True, shm_bytes)
        except:
            self._epoch_shm = SharedMemory(shm_name)
        self._epoch_arr = np.ndarray(1, buffer=self._epoch_shm.buf, dtype=np.int64)
        self._epoch_arr[0] = 0

    def __len__(self) -> int:
        """Get the length as an IterableDataset.

        Returns:
            int: Dataset length.
        """
        return self.index.get_samples_per_device()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get sample by global index.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard, index_in_shard = self.index.find_sample(index)
        reader = self.shards[shard]
        return reader[index_in_shard]

    def _get_old_sessions(self) -> Tuple[int, List[NDArray[np.int64]]]:
        """Load epoch and previous sessions from resume_shm.

        Called by __iter__() and state_dict().

        Returns:
            Tuple[int, List[NDArray[np.int64]]]: Current epoch, old sessions.
        """
        if self.resume_shm:
            text = bytes(self.resume_shm.buf).decode('utf-8')
            obj = json.loads(text)
            self._epoch_arr[0] = epoch = obj['epoch']
            old_sessions = [np.asarray(x) for x in obj['sessions']]
        else:
            epoch = int(self._epoch_arr[0])
            old_sessions = []
        return epoch, old_sessions

    def _create_cur_session(self, epoch: int) -> Tuple[NDArray[np.int64], SharedMemory]:
        """Create the current session, as we have just started an epoch.

        Called by __iter__() in a worker process, or a per-rank process if workers aren't used.

        Args:
            epoch (int): The current epoch.

        Returns:
            Tuple[NDArray[np.int64], SharedMemory]: Session and handle to shared memory.
        """
        world = World()
        shm_name = f'{self.split}_session_{epoch}'
        shm_bytes = world.num_workers * np.int64().nbytes
        try:
            shm = SharedMemory(shm_name, True, shm_bytes)
        except:
            shm = SharedMemory(shm_name)
        cur_session = np.ndarray(world.num_workers, buffer=shm.buf, dtype=np.int64)
        cur_session[:] = 0
        return cur_session, shm

    def _lookup_cur_session(
            self, epoch: int) -> Tuple[Optional[NDArray[np.int64]], Optional[SharedMemory]]:
        """Look up the current session, which exists if we are currently training.

        Called by state_dict() in a per-rank process.

        Args:
            epoch (int): The current epoch.

        Returns:
            Tuple[Optional[NDArray[np.int64]], Optional[SharedMemory]]: Maybe session, maybe shm.
        """
        shm_name = f'{self.split}_session_{epoch}'
        try:
            shm = SharedMemory(shm_name)
        except:
            return None, None
        num_workers = shm.size // np.int64().nbytes
        cur_session = np.ndarray(num_workers, buffer=shm.buf, dtype=np.int64)
        return cur_session, shm

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all the samples in our partition.

        Returns:
            Iterator[Dict[str, Any]]: Each sample.
        """
        world = World()
        epoch, old_sessions = self._get_old_sessions()
        session, session_shm = self._create_cur_session(epoch)
        sessions = old_sessions + [session]

        sequences = get_epoch(self.shard_sizes, self.shuffle, self.seed, epoch, sessions)
        todos = sequences[world.worker]

        for index in todos:
            session[world.worker] += 1
            yield self[index]

        if world.is_local_leader:
            self._epoch_arr[0] += 1

            if self.resume_shm:
                self.resume_shm.unlink()
                del self.resume_shm
                self.resume_shm = None

            session_shm.unlink()

    def _all_gather_current_session(self, session: NDArray[np.int64]) -> None:
        """All-gather the current session data.

        This is done in order to checkpoint.

        Args:
            session (NDArray[np.int64]): The current session.
        """
        # Bail if we are not multi-node.
        world = World()
        if not world.is_multinode:
            return

        # Do the all_gather on the last session counts.
        device = torch.device(f'cuda:{world.rank}')
        source = torch.tensor(session, device=device)
        dests = [
            torch.empty(len(session), dtype=torch.int64, device=device)
            for _ in range(world.num_ranks)
        ]
        dist.all_gather(dests, source)

        # Each rank provides ground truth for its workers.
        if world.is_local_leader:
            dests = torch.stack(dests).cpu().numpy()  # Shape: (world size, total workers).
            for rank in range(world.num_ranks):
                rank_start = rank * world.workers_per_rank
                rank_end = (rank + 1) * world.workers_per_rank
                session[rank_start:rank_end] = dests[rank]
        else:
            sleep(0.07)

    def state_dict(self) -> Dict[str, Any]:
        """Get a dict containing training state (called from non-worker process).

        Returns:
            Dict[str, Any]: The state.
        """
        epoch, sessions = self._get_old_sessions()
        cur_session, _ = self._lookup_cur_session(epoch)
        if cur_session is not None:
            self._all_gather_current_session(cur_session)
            sessions.append(cur_session)
        return {
            'epoch': int(epoch),
            'sessions': [x.tolist() for x in sessions],
        }

    def load_state_dict(self, obj: Dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        Args:
            obj (Dict[str, Any]): The state.
        """
        data = json.dumps(obj, sort_keys=True).encode('utf-8')
        shm_name = f'{self.split}_resume'
        try:
            self.resume_shm = SharedMemory(shm_name, True, len(data))
            self.resume_shm.buf[:] = data
        except:
            self.resume_shm = SharedMemory(shm_name)
