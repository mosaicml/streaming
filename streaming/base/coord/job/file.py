# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A Streaming job registry file."""

import json
import os
from typing import Dict, List

from typing_extensions import Self

from streaming.base.coord.job.entry import JobEntry

__all__ = ['RegistryFile']


class RegistryFile:
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
        del_job_hashes = []
        for job in filter(bool, self.jobs):
            create_time = pid2create_time.get(job.process_id)
            if not create_time or job.register_time < create_time:
                self.remove(job.job_hash)
                del_job_hashes.append(job.job_hash)
        return del_job_hashes
