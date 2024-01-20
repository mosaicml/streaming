# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""An entry in a Streaming job registry file."""

from typing import Any, Dict, List, Optional

from typing_extensions import Self

__all__ = ['JobEntry']


class JobEntry:
    """Info about a Streaming job for local dir reuse detection purposes.

    Args:
        index (int, optional): The job's index in the total list.
        job_hash (str): Job hash.
        stream_hashes (List[str]): Stream hashes.
        stream_locals (List[str], optional): Stream locals, if available.
        process_id (int): PID of local rank zero of the Streaming job.
        register_time (int): Process registration time.
    """

    def __init__(
        self,
        *,
        index: Optional[int] = None,
        job_hash: str,
        stream_hashes: List[str],
        stream_locals: Optional[List[str]] = None,
        process_id: int,
        register_time: int,
    ) -> None:
        self.index = index
        self.job_hash = job_hash
        self.stream_hashes = stream_hashes
        self.stream_locals = stream_locals
        self.process_id = process_id
        self.register_time = register_time

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
                   register_time=obj['register_time'])

    def to_json(self) -> Dict[str, Any]:
        return {
            'job_hash': self.job_hash,
            'stream_hashes': self.stream_hashes,
            # stream_locals is not saved, only their hashes.
            'process_id': self.process_id,
            'register_time': self.register_time,
        }
