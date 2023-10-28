# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A streaming WebVid dataset."""

import os
from time import sleep
from typing import Any, Dict, Optional

from streaming import StreamingDataset
from streaming.dataset import TICK, _Iterator
from streaming.storage import download_file


class StreamingInsideWebVid(StreamingDataset):
    """Streaming WebVid dataset.

    Videos are stored "inside" the shards, as is typically done.
    """

    def get_item(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().get_item(idx)
        # Processing goes here.
        return obj


class StreamingOutsideGIWebVid(StreamingDataset):
    """Streaming WebVid dataset.

    Videos are stored "outside" the shards, as a file per video. The extra download happens in
    get_item ("GI"), when samples are requested by the dataloader.

    Args:
        extra_local (str, optional): Base destination of extra local sample downloads.
        extra_remote (str, optional): Base source of extra remote sample downloads.
        **kwargs (Dict[str, Any]): Keyword arguments.
    """

    def __init__(self,
                 *,
                 extra_local: Optional[str] = None,
                 extra_remote: Optional[str] = None,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        # Videos are stored outside of their shards here.
        self.extra_local = extra_local
        self.extra_remote = extra_remote

    def get_item(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().get_item(idx)

        if self.extra_local and self.extra_remote:
            rel_path = obj['content_path']
            local = os.path.join(self.extra_local, rel_path)
            remote = os.path.join(self.extra_remote, rel_path)
            if not os.path.exists(local):
                download_file(remote, local, self.streams[0].download_timeout)
            with open(local, 'rb') as fp:
                content = fp.read()
            obj['content'] = content

        # Processing goes here.

        return obj


class StreamingOutsideDTWebVid(StreamingDataset):
    """Streaming WebVid dataset.

    Videos are stored "outside" the shards, as a file per video. The extra download happens in
    _download_thread ("DT"), when the download thread prefetches the sample.

    Args:
        extra_local (str, optional): Base destination of extra local sample downloads.
        extra_remote (str, optional): Base source of extra remote sample downloads.
        **kwargs (Dict[str, Any]): Keyword arguments.
    """

    def __init__(self,
                 extra_local: Optional[str] = None,
                 extra_remote: Optional[str] = None,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        # Videos are stored outside of their shards here.
        self.extra_local = extra_local
        self.extra_remote = extra_remote

    def get_item(self, idx: int) -> Any:
        """Get the sample at the index.

        Args:
            idx (int): Sample index.

        Returns:
            Any: The sample.
        """
        obj = super().get_item(idx)

        if self.extra_local and self.extra_remote:
            rel_path = obj['content_path']
            local = os.path.join(self.extra_local, rel_path)
            remote = os.path.join(self.extra_remote, rel_path)
            if not os.path.exists(local):
                download_file(remote, local, self.streams[0].download_timeout)
            with open(local, 'rb') as fp:
                content = fp.read()
            obj['content'] = content

        # Processing goes here.

        return obj

    def _download_thread(self, it: _Iterator) -> None:
        """Download the relevant shards in the background while we are being iterated.

        This thread is started at the beginning of each epoch, and exits either when out of samples
        or when a new epoch is started, calling exit_threads() on its state (only one epoch is
        valid at a time).

        Each worker has its own download thread, which iterates ahead of the ready thread and yield
        loop.

        Args:
            it (_Iterator): State of __iter__.
        """
        # Download loop.
        while True:
            # If we've started a new epoch early (__iter__ was called again), exit this thread
            # because there can only be one epoch at once.
            if it.should_exit():
                break

            # If we're out of samples this epoch, exit this thread because we are done downloading.
            if it.prepare_index == it.total:
                break

            # If we are requested to only pre-download so many samples, if we have as many or more
            # downloaded already, we wait and check again later.
            if self.predownload is not None:
                samples_ahead = it.prepare_index - it.yield_index
                if self.predownload <= samples_ahead:
                    sleep(TICK)
                    continue

            # If we hit -1, we skip.
            sample_id = it.sample_ids[it.prepare_index]
            if sample_id == -1:
                it.prepare_index += 1
                continue

            # Download and decompress the shard for this sample, if not already done.
            shard_id, _ = self.spanner[sample_id]
            self.prepare_shard(shard_id, False)

            # Predownload the sample's extra data.
            obj = super().get_item(sample_id)
            if self.extra_local and self.extra_remote:
                rel_path = obj['content_path']
                local = os.path.join(self.extra_local, rel_path)
                remote = os.path.join(self.extra_remote, rel_path)
                if not os.path.exists(local):
                    download_file(remote, local, self.streams[0].download_timeout)

            # Step forward one sample.
            it.prepare_index += 1

        # Note that we exited.
        it.on_exit()
