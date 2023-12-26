# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming job registry: local dir reuse detection."""

import json
import os
from hashlib import sha3_224
from shutil import rmtree
from time import sleep
from typing import Any, Dict, List, Optional, Sequence, Tuple

from filelock import FileLock
from psutil import process_iter
from typing_extensions import Self

from streaming.base.stream import Stream
from streaming.base.world import World

__all__ = ['JobRegistry', 'JobDir']


class JobEntry:
    """Info about a Streaming job for local dir reuse detection purposes.

    Args:
        index (int, optional): The job's index in the total list.
        job_hash (str): Job hash.
        stream_hashes (List[str]): Stream hashes.
        stream_locals (List[str], optional): Stream locals, if available.
        process_id (int): PID of local rank zero of the Streaming job.
        create_time (int): Process creation time.
    """

    def __init__(
        self,
        *,
        index: Optional[int] = None,
        job_hash: str,
        stream_hashes: List[str],
        stream_locals: Optional[List[str]] = None,
        process_id: int,
        create_time: int,
    ) -> None:
        self.index = index
        self.job_hash = job_hash
        self.stream_hashes = stream_hashes
        self.stream_locals = stream_locals
        self.process_id = process_id
        self.create_time = create_time

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> Self:
        """Load from JSON.

        Args:
            obj (Dict[str, Any]): Source JSON object.

        Returns:
            Self: Loaded JobEntry.
        """
        return cls(job_hash=obj['job_hash'],
                   stream_hashes=obj['stream_hashes'],
                   stream_locals=obj.get('stream_locals'),
                   process_id=obj['process_id'],
                   create_time=obj['create_time'])

    def to_json(self) -> Dict[str, Any]:
        return {
            'job_hash': self.job_hash,
            'stream_hashes': self.stream_hashes,
            # stream_locals is not saved, only their hashes.
            'process_id': self.process_id,
            'create_time': self.create_time,
        }


class JobRegistryFile:
    """StreamingDataset job registry, which is backed by a JSON file.

    Args:
        jobs (List[JobEntry]): List of StreamingDataset jobs.
    """

    def __init__(self, jobs: List[JobEntry]) -> None:
        self.jobs = []
        self.job_hash2job = {}
        self.stream_hash2job = {}
        self.num_jobs = 0
        for job in jobs:
            self.add(job)

    @classmethod
    def read(cls, filename: str) -> Self:
        if os.path.exists(filename):
            obj = json.load(open(filename))
        else:
            obj = {}
        jobs = obj.get('jobs') or []
        jobs = [JobEntry.from_json(job) for job in jobs]
        return cls(jobs)

    def write(self, filename: str) -> None:
        jobs = [job.to_json() for job in filter(bool, self.jobs)]
        obj = {'jobs': jobs}
        with open(filename, 'w') as out:
            json.dump(obj, out)

    def __len__(self) -> int:
        """Get the number of jobs registered.

        Returns:
            int: Number of registered jobs.
        """
        return self.num_jobs

    def add(self, job: JobEntry) -> None:
        """Register a Stremaing job.

        Args:
            job (Job): The job.
        """
        # Check that stream locals line up.
        if job.stream_locals:
            if len(job.stream_hashes) != len(job.stream_locals):
                raise ValueError(f'If locals are provided, must have one local per stream hash, ' +
                                 f'but got: {len(job.stream_hashes)} hashes vs ' +
                                 f'{len(job.stream_locals)} locals.')
            norm_stream_locals = job.stream_locals
        else:
            norm_stream_locals = [None] * len(job.stream_hashes)

        # Check dataset hash for reuse.
        if job.job_hash in self.job_hash2job:
            if job.stream_locals:
                raise ValueError(f'Reused dataset local path(s): {job.stream_locals}.')
            else:
                raise ValueError(f'Reused dataset local path(s): stream hashes = ' +
                                 f'{job.stream_hashes}, dataset hash = {job.job_hash}.')

        # Check each stream hash for reuse.
        for stream_hash, norm_stream_local in zip(job.stream_hashes, norm_stream_locals):
            if stream_hash in self.stream_hash2job:
                if norm_stream_local:
                    raise ValueError('Reused stream local path: {norm_stream_local}.')
                else:
                    raise ValueError('Reused stream local path: stream hash = {stream_hash}.')

        # Do the insertion.
        job.index = len(self.jobs)
        self.jobs.append(job)
        self.job_hash2job[job.job_hash] = job
        for stream_hash in job.stream_hashes:
            self.stream_hash2job[stream_hash] = job
        self.num_jobs += 1

    def remove(self, job_hash: str) -> None:
        """Deregister a Streaming job.

        Args:
            job_hash (str): Job hash.
        """
        job = self.job_hash2job.get(job_hash)
        if not job:
            raise ValueError(f'Job hash not found: {job_hash}.')

        if job.index is None:
            raise ValueError('Internal error in job registration: job index is missing.')

        self.jobs[job.index] = None
        del self.job_hash2job[job.job_hash]
        for stream_hash in job.stream_hashes:
            del self.stream_hash2job[stream_hash]
        self.num_jobs -= 1

    def filter(self, pid2create_time: Dict[int, int]) -> List[str]:
        """Filter our collection of Streaming jobs.

        Args:
            pid2create_time (Dict[int, int]): Mapping of pid to creation time.

        Returns:
            List[str]: List of hashes of removed datasets.
        """
        job_hashes = []
        for job in filter(bool, self.jobs):
            if job.create_time != pid2create_time.get(job.process_id):
                self.remove(job.job_hash)
                job_hashes.append(job.job_hash)
        return job_hashes


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
        self._filelock_filename = os.path.join(config_root, 'filelock.bin')
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
            with FileLock(self._filelock_filename):
                if os.path.exists(dirname):
                    break
            sleep(self._tick)

    def _wait_for_removal(self, job_hash: str) -> None:
        """Wait for a directory to be removed.

        Args:
            job_hash (str): Job hash of directory.
        """
        dirname = os.path.join(self.config_root, job_hash)
        while True:
            with FileLock(self._filelock_filename):
                if not os.path.exists(dirname):
                    break
            sleep(self._tick)

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
                         create_time=create_time)

        with FileLock(self._filelock_filename):
            reg = JobRegistryFile.read(self._registry_filename)
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

        with FileLock(self._filelock_filename):
            reg = JobRegistryFile.read(self._registry_filename)
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
            pass
        self._wait_for_removal(job_hash)


class JobDir:
    """Represents a Streaming job lease. On ``__del__``, cleans up after itself.

    When it goes out of scope naturally, this Job will delete its config dir and its hold on all
    the local dirs it is streaming to.

    If this process dies badly and the destructor is not reached, the same cleanup will be done by
    some future process incidentally as it registers or unregisters a Streaming job. It can tell it
    died by a combination of pid and process create time.

    Args:
        registry (JobRegistry): Stremaing job registry.
    """

    def __init__(self, registry: JobRegistry, streams: Sequence[Stream], world: World) -> None:
        self.registry = registry
        self.streams = streams
        self.world = world
        self.job_hash = registry.register(streams, world)

    def get_filename(self, path: str) -> str:
        """Get a filename by relative path under its job dir.

        Args:
            path (str): Path relative to job dir.

        Returns:
            str: Filename.
        """
        return os.path.join(self.registry.config_root, self.job_hash, path)

    def __del__(self) -> None:
        """Destructor."""
        self.registry.unregister(self.job_hash, self.world)
