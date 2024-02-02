# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A dataset, or sub-dataset if mixing, from which we stream/cache samples."""

import json
import os
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from streaming.format import shard_from_json
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
        self.got_index_size: int
        self.index: JSONDict
        self.shards: List[Shard]

    def download_index(self) -> None:
        """Download and/or verify the index file.

        Notes:
          * This method is called by StreamingDataset init.
          * It is only called in local rank zero processes.
          * This method is executed in parallel, with one Python thread per Stream, by a
            ThreadPoolExecutor.
          * It doesn't redownload if it already exists, but it still checks size/hashes.
        """
        self.got_index_size = smart_download_file(
            remote=self.remote_index_path,
            local=self.local_index_path,
            timeout=self.download_timeout,
            retry=self.download_retry,
            size=self.index_size,
            max_size=self.download_max_size,
            hashes=self.index_hashes,
            check_hashes=self.check_hashes,
        )

    def await_index(self) -> None:
        """Wait for the index file to become downloaded.

        Notes:
          * This method is called by StreamingDataset init.
          * As the index download threads run in the background, all ranks loop over all Streams,
            calling ``await_index()`` and ``load_index()`` on each Stream.
          * This method is on the critical path.
        """
        wait_for_creation(self.local_index_path, self.download_timeout, 0.07)

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
        # Read the index file and parse its JSON.
        with open(self.local_index_path) as file:
            self.index = json.load(file)

        # Create the manager-accessor for each shard according to its shard metadata in the index.
        self.shards = []
        for sub in self.index['shards']:
            shard = shard_from_json(self, sub)
            shard.validate()
            self.shards.append(shard)

        return self.shards

    def inventory_local(self, du_per_shard: NDArray[np.int64]) -> None:
        """Bring this Stream's local directory into a consistent state, tallying disk usage.

        We list all the files under the local dir up front for performance reasons, to avoid
        hammering the filesystem separately for each of potentially millions of shards, multiplied
        by files per shard, multiplied by phases of files present.

        Args:
            du_per_shard (NDArray[np.int64]): Disk usage per shard of this Stream.
        """
        local_dir = os.path.join(self.local, self.split or '')
        listing = set()
        for parent_dir, _, child_files in os.walk(local_dir):
            for child_file in child_files:
                filename = os.path.join(parent_dir, child_file)
                listing.add(filename)

        for idx, shard in enumerate(self.shards):
            du_per_shard[idx] = shard.inventory_local(listing)
