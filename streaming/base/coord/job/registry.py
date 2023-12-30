# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A directory containing all Streaming-wide filesystem state.

Useful for detecting collisions between different jobs' local dirs.
"""

import os
from hashlib import sha3_224
from shutil import rmtree
from time import sleep, time_ns
from typing import Dict, List, Sequence, Tuple

from psutil import process_iter

from streaming.base.coord.file import SoftFileLock
from streaming.base.coord.job.entry import JobEntry
from streaming.base.coord.job.file import JobFile
from streaming.base.coord.world import World
from streaming.base.stream import Stream

__all__ = ['JobRegistry']


class JobRegistry:
    """StreamingDataset job registry, for the purpose of detecting local dir reuse.

    This class is safe for concurrent access via a filelock.

    Args:
        config_root (str): Streaming configuration root directory, used for collision detection,
            filelock paths, etc. Defaults to ``/tmp/streaming``, using the equivalent temp root on
            your system.
    """

    def __init__(self, config_root: str, tick: float = 0.007) -> None:
        self.config_root = config_root
        self._tick = tick
        self._lock_filename = os.path.join(config_root, 'registry.lock')
        self._lock = SoftFileLock(self._lock_filename)
        self._registry_filename = os.path.join(config_root, 'registry.json')

    def _get_live_procs(self) -> Dict[int, int]:
        """List the pids and creation times of every live process in the system.

        The creation times protect us from PID reuse.

        Returns:
            Dict[int, int]: Mapping of pid to integer creation time.
        """
        ret = {}
        for obj in process_iter(['pid', 'create_time']):
            ret[obj.pid] = int(obj.create_time() * 1e9)
        return ret

    def _hash(self, data: bytes) -> str:
        """Get a short, deterministic, fixed-length code for the given data.

        Args:
            data (bytes): The data to hash.

        Returns:
            str: Truncated hex digest.
        """
        return sha3_224(data).hexdigest()[:8]

    def _hash_streams(self, streams: Sequence[Stream]) -> Tuple[List[str], List[str], str]:
        """Get a short, opaque str key for a StreamingDataset and each of its Streams.

        This is useful for collision detection.

        Args:
            streams (Sequence[Stream]): List of this StreamingDataset's Streams, which in
                combination with process IDs and creation times lets us uniquely identify a
                Streaming job.

        Returns:
            Tuple[str, List[str], List[str]]: Triple of (normalized stream locals, stream hashes,
                and dataset hash).
        """
        # Get a list of the normalized locals of each Stream.
        stream_locals = []
        for stream in streams:
            local = os.path.join(stream.local, stream.split)
            local = os.path.normpath(local)
            local = os.path.abspath(local)
            stream_locals.append(local)

        # Collect the locals into a deduped set.
        stream_locals_set = set()
        for local in stream_locals:
            if local in stream_locals_set:
                raise ValueError(f'Reused local path: {local}.')
            stream_locals_set.add(local)

        # Verify that no local is contained within another local.
        for local in stream_locals:
            parts = local.split(os.path.sep)[1:]
            for num_parts in range(1, len(parts) - 1):  # Leftmost is '' because they start with /.
                parent = os.path.sep.join(parts[:num_parts])
                if parent in stream_locals_set:
                    raise ValueError(f'One local path contains another local path: {parent} vs ' +
                                     f'{local}.')

        # Hash each local.
        stream_hashes = []
        for local in sorted(stream_locals):
            data = local.encode('utf-8')
            stream_hash = self._hash(data)
            stream_hashes.append(stream_hash)

        # Hash the dataset.
        text = ','.join(stream_hashes)
        data = text.encode('utf-8')
        job_hash = self._hash(data)

        return stream_locals, stream_hashes, job_hash

    def _make_dir(self, job_hash: str) -> None:
        """Create a Streaming job config dir.

        Args:
            job_hash: Streaming config subdir for this job.
        """
        dirname = os.path.join(self.config_root, job_hash)
        os.makedirs(dirname)

    def _remove_dir(self, job_hash: str) -> None:
        """Delete a Streaming job config dir.

        Args:
            job_hash: Streaming config subdir for this job.
        """
        dirname = os.path.join(self.config_root, job_hash)
        rmtree(dirname)

    def _wait_for_existence(self, job_hash: str) -> None:
        """Wait for a directory to be created.

        Args:
            job_hash (str): Job hash of directory.
        """
        dirname = os.path.join(self.config_root, job_hash)
        while True:
            sleep(self._tick)
            with self._lock:
                if os.path.exists(dirname):
                    break

    def _wait_for_removal(self, job_hash: str) -> None:
        """Wait for a directory to be removed.

        Args:
            job_hash (str): Job hash of directory.
        """
        dirname = os.path.join(self.config_root, job_hash)
        while True:
            sleep(self._tick)
            with self._lock:
                if not os.path.exists(dirname):
                    break

    def _register(self, streams: Sequence[Stream]) -> str:
        """Register this collection of StreamingDataset replicas.

        Called by the local leader.

        Args:
            streams (Sequence[Stream]): List of this StreamingDataset's Streams, which in
                combination with process IDs and creation times lets us uniquely identify a
                Streaming job.

        Returns:
            str: Streaming config subdir for this job.
        """
        register_time = time_ns()
        pid2create_time = self._get_live_procs()
        pid = os.getpid()
        create_time = pid2create_time.get(pid)
        if create_time is None:
            raise RuntimeError('`psutil` thinks we are dead, and yet here we are: pid = {pid}.')

        stream_locals, stream_hashes, job_hash = self._hash_streams(streams)

        entry = JobEntry(job_hash=job_hash,
                         stream_hashes=stream_hashes,
                         stream_locals=stream_locals,
                         process_id=pid,
                         register_time=register_time)

        with self._lock:
            reg = JobFile.read(self._registry_filename)
            reg.add(entry)
            del_job_hashes = reg.filter(pid2create_time)
            reg.write(self._registry_filename)
            map(self._remove_dir, del_job_hashes)
            self._make_dir(job_hash)

        return job_hash

    def _lookup(self, streams: Sequence[Stream]) -> str:
        """Look up this collection of StreamingDataset replicas.

        Called by the local leader.

        Args:
            streams (Sequence[Stream]): List of this StreamingDataset's Streams, which in
                combination with process IDs and creation times lets us uniquely identify a
                Streaming job.

        Returns:
            str: Streaming config subdir for this job.
        """
        _, _, job_hash = self._hash_streams(streams)
        return job_hash

    def register(self, streams: Sequence[Stream], world: World) -> str:
        """Register or look up this collection of StreamingDataset replicas.

        Called by all ranks.

        Args:
            streams (Sequence[Stream]): List of this StreamingDataset's Streams, which in
                combination with process IDs and creation times lets us uniquely identify a
                Streaming job.
            world (World): Rank-wise world state.

        Returns:
            str: Subdir for this collection of StreamingDataset replicas.
        """
        if world.is_local_leader:
            job_hash = self._register(streams)
        else:
            job_hash = self._lookup(streams)
            self._wait_for_existence(job_hash)
        return job_hash

    def _unregister(self, job_hash: str) -> None:
        """Unregister this collection of StreamingDataset replicas.

        Called by the local leader.

        Args:
            job_hash (str): Subdir identifying this Streaming job.
        """
        pid2create_time = self._get_live_procs()

        with self._lock:
            reg = JobFile.read(self._registry_filename)
            reg.remove(job_hash)
            del_job_hashes = reg.filter(pid2create_time)
            reg.write(self._registry_filename)
            map(self._remove_dir, del_job_hashes)
            self._remove_dir(job_hash)

    def unregister(self, job_hash: str, world: World) -> None:
        """Unregister this collection of StreamingDataset replicas.

        Called by all ranks.

        Args:
            job_hash (str): Subdir identifying this Streaming job.
            world (World): Rank-wise world state.
        """
        if world.is_local_leader:
            self._unregister(job_hash)
        else:
            self._wait_for_removal(job_hash)
