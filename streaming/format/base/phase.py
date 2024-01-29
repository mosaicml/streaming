# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""An individual file of a Streaming shard."""

import os
from typing import Any, Dict, Optional, Set

from typing_extensions import Self

from streaming.format.base.stream_conf import StreamConf
from streaming.hashing import get_hash
from streaming.storage.download import download_file
from streaming.util.retrying import retry

__all__ = ['ShardFilePhase']


class ShardFilePhase:
    """Metadata for validating one specific phase of a file.

    Args:
        stream (StreamConf): Link back up to the Stream that owns this shard, from which we get
            arguments which are shared across all shards like remote/local paths. Avoids an import
            cycle by Stream subclassing StreamConf.
        relative_path (str): Dataset-relative path to file.
        size  (int, optional): Size, if known in advance. Defaults to ``None``.
        hashes (Dict[str, str], optional): Hashes, if known in advance. Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        stream: StreamConf,
        relative_path: str,
        size: Optional[int] = None,
        hashes: Optional[Dict[str, str]] = None,
    ) -> None:
        if size is not None and size < 0:
            raise ValueError(f'Shard file size must be a non-negative integer, but got: {size}.')

        self.stream = stream
        self.relative_path = relative_path
        self.size = size
        self.hashes = hashes or {}

    @classmethod
    def from_json(cls, stream: StreamConf, obj: Dict[str, Any]) -> Self:
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
            stream (StreamConf): Reference to the owning Stream.
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

    def set_stream(self, stream: StreamConf) -> None:
        """Save a link to the owning Stream, as many Stream args apply to all its Shards.

        Args:
            stream (StreamConf): The Stream that owns this Shard.
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

        if self.stream.download_max_size < self.size:
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

    def init_dir(self, listing: Set[str]) -> int:
        """Initialize the given local directory wrf this shard file phase.

        Args:
            listing (Set[str]): Recursive dataset file listing.

        Returns:
            int: Disk usage.
        """
        filename = self.get_local_filename()
        if filename not in listing:
            return 0

        if self.size is not None:
            try:
                got_size = os.stat(filename).st_size
            except:
                return 0
            if got_size == self.size:
                return self.size
            else:
                os.remove(filename)
                return 0
        else:
            if self.stream.allow_unchecked_resumption:
                try:
                    return os.stat(filename).st_size
                except:
                    return 0
            else:
                os.remove(filename)
                return 0

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
        local = self.get_local_filename()
        remote = self.get_remote_filename()
        download = lambda: download_file(remote, local, self.stream.download_timeout)
        retry(num_attempts=self.stream.download_retry)(download)
        size = os.stat(local).st_size

        # Validate downloaded size agsainst expected size.
        if self.size is not None:
            if self.size != size:
                raise ValueError(f'Downloaded file was not the expected size: expected ' +
                                 f'{self.size:,} bytes but got {size:,} bytes.')

        # Validate downloaded size against limit.
        #
        # This check is necessary when expected size is not known, otherwise expected size vs limit
        # in validate() and downloaded size vs expected size above has it covered, but it's
        # basically free, so let's be extra careful.
        if self.stream.download_max_size is not None:
            if self.stream.download_max_size < size:
                raise ValueError(f'Download was too large: {self.get_local_filename()} is ' +
                                 f'{size:,} bytes vs limit {self.stream.download_size} ' +
                                 f'bytes. As you raise shard size, you will experience ' +
                                 f'hard-to-debug choppiness, thrashing, and indefinite stalls. ' +
                                 f'Please reduce shard size. To continue anyway, raise the ' +
                                 f'StreamingDataset or Stream argument `download_size` or set ' +
                                 f'it to `None` to disable this check completely.')

        # Validate hashes against expected.
        if self.stream.apply_hash_algos:
            data = open(local, 'rb').read()
            for algo in self.stream.apply_hash_algos:
                if algo in self.hashes:
                    if get_hash(algo, data) == self.hashes[algo]:
                        break
                    else:
                        raise ValueError(f'Hash check failure: {local}.')
            else:
                raise ValueError(f'There is no overlap between what hash algorithms we have ' +
                                 f'chosen to validate shard files with and what hash algos ' +
                                 f'and digests the index has stored for us to compare against: ' +
                                 f'we wanted {self.stream.apply_hash_algos}, but the index ' +
                                 f'accepts {sorted(self.hashes)}.')

        return size

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
