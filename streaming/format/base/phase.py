# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""An individual file of a Streaming shard."""

import os
from shutil import rmtree
from typing import Any, Dict, Optional, Set

from typing_extensions import Self

from streaming.storage.extra import smart_download_file
from streaming.stream.dir_conf import StreamDirConf

__all__ = ['ShardFilePhase']


class ShardFilePhase:
    """Metadata for validating one specific phase of a file.

    Args:
        stream (StreamDirConf): Link back up to the Stream that owns this shard, from which we get
            arguments which are shared across all shards like remote/local paths. Avoids an import
            cycle by Stream subclassing StreamDirConf.
        relative_path (str): Dataset-relative path to file.
        size (int, optional): Size, if known in advance. Defaults to ``None``.
        hashes (Dict[str, str], optional): Hashes, if known in advance. Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        stream: StreamDirConf,
        relative_path: str,
        size: Optional[int] = None,
        hashes: Optional[Dict[str, str]] = None,
    ) -> None:
        # Checks.
        if size is not None and size < 0:
            raise ValueError(f'Shard file size must be a non-negative integer, but got: {size}.')

        # Provided to init.
        self.stream = stream
        self.relative_path = relative_path
        self.size = size
        self.hashes = hashes or {}

        # Not known until we `download()` or `inventory_local()`.
        self.got_size: Optional[int]

    @classmethod
    def from_json(cls, stream: StreamDirConf, obj: Dict[str, Any]) -> Self:
        """Initialize from JSON object.

        Example input:

        ```json
            {
              "basename": "shard.00000.mds",
              "bytes": 1048544,
              "hashes": {
                "sha1": "8d0634d3836110b00ae435bbbabd1739f3bbeeac",
                "xxh3_64": "2c54988514bca807"
              }
            }
        ```

        Args:
            stream (StreamDirConf): Reference to the owning Stream.
            obj (Dict[str, Any]): Shard file phase JSON metadata.

        Returns:
            Self: The loaded MDS shard object.
        """
        return cls(
            stream=stream,
            relative_path=obj['basename'],
            size=obj['bytes'],
            hashes=obj['hashes'],
        )

    def set_stream(self, stream: StreamDirConf) -> None:
        """Save a link to the owning Stream, as many Stream args apply to all its Shards.

        Args:
            stream (StreamDirConf): The Stream that owns this Shard.
        """
        self.stream = stream

    def validate_for_download(self) -> None:
        """Validate for download purposes.

        The download phase (``zip`` or ``raw``) of a shard file must be sized, and if there is a
        limit on download size, must fit the limit.
        """
        if self.size is None:
            raise ValueError(
                f'The first existing phase, i.e. the phase that is stored persistently and ' +
                f'downloaded, must be of known size.')

        if self.stream.download_max_size is not None and self.stream.download_max_size < self.size:
            raise ValueError(
                f'Download would be too large: {self.get_remote_filename()} is {self.size:,} ' +
                f'bytes vs limit {self.stream.download_max_size} bytes. As you raise shard size, '
                + f'you will experience hard-to-debug choppiness, thrashing, and indefinite ' +
                f'stalls. Please reduce shard reduce shard size. To continue anyway, raise the ' +
                f'StreamingDataset or Stream argument `download_max_size` or set it to `None` ' +
                f'to disable this check completely.')

    def get_local_filename(self) -> str:
        """Get this phase's local filename.

        Returns:
            str: Local filename.
        """
        return os.path.join(self.stream.local, self.stream.split or '', self.relative_path)

    def get_remote_filename(self) -> str:
        """Get this phase's remote filename.

        Returns:
            str: Remote filename or path.
        """
        if not self.stream.remote:
            raise ValueError('Asked for a remote path, but remote does not exist.')
        return os.path.join(self.stream.remote, self.stream.split or '', self.relative_path)

    def probe(self, listing: Set[str]) -> bool:
        """Probe the given directory listing for this file phase.

        Try to minimize hitting the filesystem, for performance resaons.

        Args:
            listing (Set[str]): Recursive dataset file listing.

        Returns:
            bool: Whether this phase is present or not in the listing.
        """
        filename = self.get_local_filename()
        return filename in listing

    def inventory_local(self, listing: Set[str]) -> Optional[int]:
        """Initialize the given local directory wrf this shard file phase.

        Args:
            listing (Set[str]): Recursive dataset file listing.

        Returns:
            int: Disk usage.
        """
        filename = self.get_local_filename()

        # Is the file in the listing?
        if filename not in listing:
            # Problem: it's not listed.
            got_size = None
        else:
            # It's listed. Is the file actually there in the filesystem?
            if not os.path.exists(filename):
                # Problem: it's not there anymore.
                got_size = None
            elif not os.path.isfile(filename):
                # Problem: it's not a file.
                rmtree(filename)
                got_size = None
            else:
                # It's there. Do we know its expected size?
                if self.size is not None:
                    # Expected size is known, so they had better match.
                    got_size = os.stat(filename).st_size
                    if got_size != self.size:
                        os.remove(filename)
                        got_size = None
                else:
                    # Expected size is unknown. Is that a problem?
                    if self.stream.allow_unchecked_resumption:
                        # Not a problem.
                        got_size = os.stat(filename).st_size
                    else:
                        # Is a problem.
                        os.remove(filename)
                        got_size = None

        self.got_size = got_size
        return got_size

    def is_local(self) -> bool:
        """Check the filesystem for this phase of the file.

        Returns:
            bool: Whether this phase is local.
        """
        filename = self.get_local_filename()
        return os.path.isfile(filename)

    def download(self) -> int:
        """Download this phase.

        Returns:
            int: Delta disk usage, in bytes.
        """
        # Verify there is a remote to download from.
        if self.stream.remote in {None, self.stream.local}:
            raise ValueError(f'File(s) are missing, but there is nowhere to download them from ' +
                             f'because we are the authoritative copy.')

        # Do the download, retrying if necessary.
        remote = self.get_remote_filename()
        local = self.get_local_filename()

        self.got_size = smart_download_file(
            remote=remote,
            local=local,
            timeout=self.stream.download_timeout,
            retry=self.stream.download_retry,
            size=self.size,
            max_size=self.stream.download_max_size,
            hashes=self.hashes,
            check_hashes=self.stream.check_hashes,
        )

        return self.got_size

    def evict(self) -> int:
        """Delete this phase.

        Returns:
            int: Delta disk usage, in bytes.
        """
        ddu = 0
        filename = self.get_local_filename()
        size = self.size if self.size is not None else os.stat(filename).st_size
        if os.path.isfile(filename):
            os.remove(filename)
            ddu -= size
        return ddu
