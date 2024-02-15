# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A directory containing all dataset-wide filesystem state for a Streaming job."""

import os
from typing import Sequence

from streaming.base.coord.job.registry import JobRegistry
from streaming.base.stream import Stream
from streaming.base.world import World

__all__ = ['JobDir']


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

    def manual_unregister(self) -> None:
        """Explicitly un-register the job ahead of its deletion.

        This is useful when you want to ensure that this job is un-registered synchronously instead
        of whenever the garbage collector eventually gets around to it.

        This job must be registered when this is called.
        """
        self.registry.unregister(self.job_hash, self.world, True)

    def __del__(self) -> None:
        """Destructor.

        You may unregister the job explicitly ahead of time (to ensure it happens synchronously
        instead of eventually).
        """
        self.registry.unregister(self.job_hash, self.world, False)
