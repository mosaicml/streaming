# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A directory containing all Streaming-wide filesystem state.

Useful for detecting collisions between different jobs' local dirs.
"""

import os
from hashlib import sha3_224
from shutil import rmtree
from time import time_ns
from typing import Dict, List, Optional, Sequence, Tuple

from filelock import FileLock
from psutil import process_iter

from streaming.base.coord.filesystem.waiting import wait_for_creation, wait_for_deletion
from streaming.base.coord.job.entry import JobEntry
from streaming.base.coord.job.file import RegistryFile
from streaming.base.stream import Stream
from streaming.base.world import World

__all__ = ['JobRegistry']


class JobRegistry:
    """StreamingDataset job registry, for the purpose of detecting local dir reuse.

    This class is safe for concurrent access via a filelock.

    Args:
        config_root (str): Streaming configuration root directory, used for collision detection,
            filelock paths, etc. Defaults to ``/tmp/streaming``, using the equivalent temp root on
            your system.
        timeout (float, optional): How long to wait before raising an exception, in seconds.
            Defaults to ``30``.
        tick (float): Check interval, in seconds. Defaults to ``0.007``.
    """

    def __init__(
        self,
        config_root: str,
        timeout: Optional[float] = 30,
        tick: float = 0.007,
    ) -> None:
        self.config_root = config_root
        self.timeout = timeout
        self.tick = tick

        self.lock_filename = os.path.join(config_root, 'registry.lock')
        self.lock = FileLock(self.lock_filename)

        self.registry_filename = os.path.join(config_root, 'registry.json')

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

    def _make_job_dir(self, job_hash: str) -> None:
        """Create a Streaming job config dir.

        Args:
            job_hash: Streaming config subdir for this job.
        """
        dirname = os.path.join(self.config_root, job_hash)
        os.makedirs(dirname)

    def _remove_job_dir(self, job_hash: str) -> None:
        """Delete a Streaming job config dir.

        Args:
            job_hash: Streaming config subdir for this job.
        """
        dirname = os.path.join(self.config_root, job_hash)
        rmtree(dirname)

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
        if not world.is_local_leader:
            _, _, job_hash = self._hash_streams(streams)
            dirname = os.path.join(self.config_root, job_hash)
            wait_for_creation(dirname, self.timeout, self.tick, self.lock)
            return job_hash

        # Collect our stream locals and hash them, resulting in a job hash.
        stream_locals, stream_hashes, job_hash = self._hash_streams(streams)

        with self.lock:
            # Get registration time.
            register_time = time_ns()

            # Load the job database.
            db = RegistryFile.read(self.registry_filename)

            # Perform liveness checks on the jobs we have registered.
            pid2create_time = self._get_live_procs()
            del_job_hashes = db.filter(pid2create_time)

            # Add an entry for this job.
            pid = os.getpid()
            create_time = pid2create_time.get(pid)
            if create_time is None:
                raise RuntimeError('`psutil` thinks we are dead, and yet here we are: pid {pid}.')
            entry = JobEntry(job_hash=job_hash,
                             stream_hashes=stream_hashes,
                             stream_locals=stream_locals,
                             process_id=pid,
                             register_time=register_time)
            db.add(entry)

            # Save the new db to disk.
            db.write(self.registry_filename)

            # Add and remove job directories accordingly.
            self._make_job_dir(job_hash)
            map(self._remove_job_dir, del_job_hashes)

            return job_hash

    def is_registered(self, job_hash: str) -> bool:
        """Tell whether the given job_hash is registered.

        Called by all ranks.

        Args:
            job_hash (str): Potentially registered job hash.

        Returns:
            bool: Whether the job hash is registered.
        """
        dirname = os.path.join(self.config_root, job_hash)
        with self.lock:
            return os.path.isdir(dirname)

    def unregister(self, job_hash: str, world: World, strict: bool = True) -> None:
        """Unregister this collection of StreamingDataset replicas.

        Called by all ranks.

        Args:
            job_hash (str): Subdir identifying this Streaming job.
            world (World): Rank-wise world state.
            strict (bool): If strict, require the job to be currently registered at start.
        """
        if not world.is_local_leader:
            dirname = os.path.join(self.config_root, job_hash)
            wait_for_deletion(dirname, self.timeout, self.tick, self.lock)
            return

        with self.lock:
            # Load the job database.
            db = RegistryFile.read(self.registry_filename)

            # Check if the job hash is registered.
            was_registered = db.contains(job_hash)

            # If strict, require the job to be registered.
            if strict and not was_registered:
                raise ValueError(f'Attempted to unregister job {job_hash}, but it was not ' +
                                 f'registered.')

            # Unregister the job, if it is registered.
            if was_registered:
                db.remove(job_hash)
                self._remove_job_dir(job_hash)

            # Perform liveness checks on the jobs we have registered.
            pid2create_time = self._get_live_procs()
            del_job_hashes = db.filter(pid2create_time)

            # If we unregistered the job and/or we garbage collected job(s), save the new jobs
            # database back to disk.
            if was_registered or del_job_hashes:
                db.write(self.registry_filename)

            # Remove each directory corresponding to a job that was garbage collected.
            map(self._remove_job_dir, del_job_hashes)
