# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A dataset, or sub-dataset if mixing, from which we stream/cache samples."""

import json
import os
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from streaming.constant import FILESYSTEM_POLL_INTERVAL
from streaming.format.base.shard.base import Shard
from streaming.storage.extra import smart_download_file
from streaming.stream.dir_conf import StreamDirConf
from streaming.stream.weight_conf import StreamWeightConf
from streaming.util.json import JSONDict
from streaming.util.waiting import wait_for_creation

__all__ = ['Stream']


class Stream(StreamDirConf, StreamWeightConf):
    """A StreamingDatasdet directory, used alone or together with others.

    Args:
        kwargs (Dict[str, Any]): Arguments inherted from StreamDirConf and StreamWeightConf.
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        StreamDirConf.__init__(self, **kwargs)
        StreamWeightConf.__init__(self, **kwargs)
        self.index_size: int
        self.index: JSONDict
        self.shards: List[Shard]

    def download_index(self) -> None:
        """Download and/or verify the index file.

        Notes:
          * This method is called by StreamingDataset init.
          * It is only called in local rank zero processes.
          * This method is executed in parallel, with one Python process per Stream, by a
            procees pool.
          * It doesn't redownload if it already exists, but it still checks size/hashes.
        """
        try:
            # Download and/or verify the index file.
            self.index_size = smart_download_file(
                remote=self.remote_index_path,
                local=self.local_index_path,
                timeout=self.download_timeout,
                retry=self.download_retry,
                max_size=self.download_max_size,
            )
        except Exception as err:
            # Write an empty file to signal that we are done downloading to `await_index()` (but it
            # will crash upon load).
            with open(self.local_index_path, 'wb') as out:
                out.write(b'')

            # Write the error that was raised to a shadow index file path. This file will be read
            # by other processes once they discover this index download failed.
            with open(self.local_index_path + '.error', 'w') as out:
                out.write(str(err))

            # Propagate the error within this process.
            raise err

    def await_index(self) -> None:
        """Wait for the index file to become downloaded.

        Notes:
          * This method is called by StreamingDataset init.
          * As the index download threads run in the background, all ranks loop over all Streams,
            calling ``await_index()`` and ``load_index()`` on each Stream.
          * This method is on the critical path.
        """
        # Wait for the index file to exist.
        wait_for_creation(self.local_index_path, self.download_timeout, FILESYSTEM_POLL_INTERVAL)

        # Because `download_index()` was called in another process, we must set `self.index_size`.
        self.index_size = os.stat(self.local_index_path).st_size

    def load_index(self) -> List[Shard]:
        """Wait for the index file to become downloaded, then load it.

        Notes:
          * This method is called by StreamingDataset init.
          * As the index download threads run in the background, all ranks loop over all Streams,
            calling ``await_index()`` and ``load_index()`` on each Stream.
          * This method is on the critical path.

        Returns:
            List[Shared]: List of loaded Shards.
        """
        from streaming.format import shard_from_json

        # Read the index file.
        with open(self.local_index_path, 'rb') as file:
            data = file.read()

        if not data:
            path = self.local_index_path + '.error'
            if os.path.exists(path):
                with open(path, 'r') as file:
                    text = file.read()
                    raise ValueError(text)
            else:
                raise ValueError(f'Index file {self.local_index_path} is empty.')

        try:
            text = data.decode('utf-8')
            self.index = json.loads(text)
            # Create the manager-accessor for each shard according to its shard metadata in the index.
            self.shards = []
            for sub in self.index['shards']:
                shard = shard_from_json(self, sub)
                shard.validate()
                self.shards.append(shard)
        except Exception as err:
            raise ValueError(f'Index file {self.local_index_path} is corrupted: {err}.')

        return self.shards

    def inventory_local(self, cache_usage_per_shard: NDArray[np.int64]) -> None:
        """Bring this Stream's local directory into a consistent state, tallying disk usage.

        We list all the files under the local dir up front for performance reasons, to avoid
        hammering the filesystem separately for each of potentially millions of shards, multiplied
        by files per shard, multiplied by phases of files present.

        Args:
            cache_usage_per_shard (NDArray[np.int64]): Disk usage per shard of this Stream.
        """
        local_dir = os.path.join(self.local, self.split or '')
        listing = set()
        for parent_dir, _, child_files in os.walk(local_dir):
            for child_file in child_files:
                filename = os.path.join(parent_dir, child_file)
                listing.add(filename)

        for idx, shard in enumerate(self.shards):
            cache_usage_per_shard[idx] = shard.inventory_local(listing)
