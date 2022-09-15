# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from enum import IntEnum
from multiprocessing import Pool
from threading import RLock, Thread
from time import sleep
from typing import Any, Iterator, List, Optional

import numpy as np
from torch.utils.data import IterableDataset

from streaming.base import distributed as dist
from streaming.base.compression import decompress
from streaming.base.download import download_or_wait
from streaming.base.format import reader_from_json
from streaming.base.hashing import get_hash
from streaming.base.index import Index, Partition, get_index_basename


class DownloadStatus(IntEnum):
    """Download status."""
    NOT_STARTED = 1
    IN_PROGRESS = 2
    DONE = 3
    FAILED = 4


class Dataset(IterableDataset):
    """A sharded, streamed, iterable dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        remote (Optional[str]): Download shards from this remote path or directory. If None, this
            rank and worker partition of the dataset must all exist locally. Default: ``None``.
        split (Optional[str]): Which dataset split to use, if any. Default: ``None``.
        shuffle (bool): Whether to shuffle the samples while iterating. Default: ``None``.
        prefetch (Optional[int]): Target number of samples remaining to prefetch while iterating.
            Default: ``None``.
        keep_zip (Optional[bool]): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Default:
            ``None``.
        retry (int): Number of download re-attempts before giving up. Default: ``2``.
        timeout (float): Number of seconds to wait for a shard to download before raising an
            exception. Default: ``60``.
        hash (Optional[str]): Optional hash or checksum algorithm to use to validate shards.
            Default: ``None``.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device
            DataLoader. Default: ``None``.

    .. doctest::

        To write the dataset:
        >>> import numpy as np
        >>> from PIL import Image
        >>> from uuid import uuid4
        >>> from streaming import MDSWriter
        >>> dirname = 'dirname'
        >>> columns = {
        ...     'uuid': 'str',
        ...     'img': 'jpeg',
        ...     'clf': 'int'
        ... }
        >>> compression = 'zstd'
        >>> hashes = 'sha1', 'xxh64'
        >>> samples = [
        ...     {
        ...         'uuid': str(uuid4()),
        ...         'img': Image.fromarray(np.random.randint(0, 256, (32, 48, 3), np.uint8)),
        ...         'clf': np.random.randint(10),
        ...     }
        ...     for i in range(1000)
        ... ]
        >>> with MDSWriter(dirname, columns, compression, hashes) as out:
        ...     for sample in samples:
        ...         out.write(sample)

        To read the dataset:
        >>> from streaming import Dataset
        >>> dataset = Dataset(dirname)
        >>> for sample in dataset:
        ...     print(sample)

        To read the dataset (with all optional arguments):
        >>> from streaming import Dataset
        >>> dataset = Dataset(local=dirname, remote=None, split=None, shuffle=True,
        ...                   prefetch=100_000, keep_zip=None, retry=2, timeout=60, hash=None,
        ...                   batch_size=None)
    """

    def __init__(self,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = True,
                 prefetch: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 retry: int = 2,
                 timeout: float = 60,
                 hash: Optional[str] = None,
                 batch_size: Optional[int] = None) -> None:
        if keep_zip is None:
            keep_zip = remote is None or remote == local
        hash = hash or None

        self.local = local
        self.remote = remote
        self.split = split or ''
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.keep_zip = keep_zip
        self.retry = retry
        self.timeout = timeout
        self.hash = hash
        self.batch_size = batch_size

        basename = get_index_basename()
        wait = dist.get_local_rank() != 0
        filename = self._download_file(basename, wait)
        obj = json.load(open(filename))
        assert obj['version'] == 2

        self.shards = []
        for info in obj['shards']:
            shard = reader_from_json(local, split, info)
            self.shards.append(shard)

        samples_per_shard = list(map(lambda shard: shard.samples, self.shards))
        self.index = Index(samples_per_shard, batch_size)

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock: RLock
        self._has_shard = np.zeros(len(self.shards), np.uint8)
        self._next_epoch = 0
        self._epoch_to_todo_ids = {}
        self._downloaded_ids = []
        self._download_status = DownloadStatus.NOT_STARTED
        self._download_exception: Exception

    def __len__(self) -> int:
        """Get the length as an IterableDataset (ie, divided by num devices).

        Returns:
            int: Dataset length.
        """
        return self.index.get_samples_per_device()

    def _load_shards(self, shards: List[int], partition: Partition) -> None:
        """Load our partition's samples from the given locally cached shards.

        Every time you call __iter__ on this dataset, it registers the list of samples you have
        left, which will not be the full epoch if the dataset isn't finished loaded when you start
        training.

        Calls to this method during training modify the samples remaining on these iterations on
        the fly to insert these new samples and then re-sort, making the shuffle as perfect as was
        possible.

        This operation is heavy and takes the lock, so call this method with all available shards
        at once.

        Args:
            shards (List[int]): Shard IDs.
            partition (Partition): Our rank and worker's partition of the dataset.
        """
        # Get our partition and shards' sample ranges.
        new_ids = []
        for shard in shards:
            shard_min_id = max(self.index.shard_offsets[shard], partition.min_sample_id)
            shard_max_id = min(self.index.shard_offsets[shard + 1] - 1, partition.max_sample_id)
            new_ids += list(range(shard_min_id, shard_max_id + 1))

        with self._lock:
            # Extend and optionally reshuffle the remaining samples of any epochs in progress.
            if self.shuffle:
                if self._download_status == DownloadStatus.IN_PROGRESS:
                    self._downloaded_ids.extend(new_ids)
                    np.random.shuffle(self._downloaded_ids)
                for todo_ids in self._epoch_to_todo_ids.values():
                    todo_ids.extend(new_ids)
                    np.random.shuffle(todo_ids)
            else:
                if self._download_status == DownloadStatus.IN_PROGRESS:
                    self._downloaded_ids.extend(new_ids)
                for todo_ids in self._epoch_to_todo_ids.values():
                    todo_ids.reverse()
                    todo_ids.extend(new_ids)
                    todo_ids.reverse()

            # Note that we have loaded the shards.
            for shard in shards:
                self._has_shard[shard] = True

    def _load_shard(self, shard: int, partition: Partition) -> None:
        """Load our partition's samples from the given locally cached shard.

        For performance reasons, prefer _load_shards() where possible.

        Args:
            shard (int): Shard ID.
            partition (Partition): Our rank and worker's partition of the dataset.
        """
        self._load_shards([shard], partition)

    def _preload_shard(self, shard: int) -> bool:
        """Decompress and validate a single shard, returning whether present.

        Args:
            shard (int): Which shard.

        Returns:
            bool: Whether shard is present.
        """
        info = self.shards[shard]
        for raw_info, zip_info in info.file_pairs:
            raw_filename = os.path.join(self.local, self.split, raw_info.basename)
            if os.path.isfile(raw_filename):
                if self.hash:
                    data = open(raw_filename, 'rb').read()
                    assert get_hash(self.hash, data) == raw_info.hashes[self.hash]
            elif not zip_info:
                return False
            else:
                zip_filename = os.path.join(self.local, self.split, zip_info.basename)
                if os.path.isfile(zip_filename):
                    data = open(zip_filename, 'rb').read()
                    if self.hash:
                        assert get_hash(self.hash, data) == zip_info.hashes[self.hash]
                    data = decompress(info.compression, data)  # pyright: ignore
                    with open(raw_filename, 'wb') as out:
                        out.write(data)
                    # if not self.keep_zip:
                    #     os.remove(zip_filename)
                else:
                    return False
        return True

    def _preload(self, partition: Partition) -> List[int]:
        """Load any shards that are cached locally, returning missing shards.

        Args:
            partition (Partition): Our rank and worker's partition of the dataset.

        Returns:
            List[int]: Missing shards that must be downloaded.
        """
        # Create lock in preload() because we are prevented from putting it in __init__ because of
        # DataLoader num_workers and fork/spawn semantics.
        if not hasattr(self, '_lock'):
            self._lock = RLock()

        # Bail out if has already been called.
        with self._lock:
            if self._download_status != DownloadStatus.NOT_STARTED:
                return []
            self._download_status = DownloadStatus.IN_PROGRESS

        # Find and load cached shards given our sample range.
        present_shards = []
        missing_shards = []
        for shard in partition.shards:
            if self._preload_shard(shard):
                present_shards.append(shard)
            else:
                missing_shards.append(shard)
        self._load_shards(present_shards, partition)

        # If there are no missing shards, we're done.
        if not missing_shards:
            with self._lock:
                self._download_status = DownloadStatus.DONE
            return []

        # Always download the first shard first, if it is missing, because other workers may be
        # waiting on it.
        if self.shuffle:
            if missing_shards[0] == partition.shards[0]:
                nonfirst = 1
            else:
                nonfirst = 0
            missing_shards = np.array(missing_shards)
            np.random.shuffle(missing_shards[nonfirst:])
            missing_shards = missing_shards.tolist()

        return missing_shards

    def _download_file(self, basename: str, wait: bool = False) -> str:
        """Safely download a file from remote to local cache.

        Args:
            basename (str): Basename of file to download.
            wait (bool): Whether to wait for another worker to download the file.

        Returns:
            str: Local cache filename.
        """
        if self.remote is None:
            remote = None
        else:
            remote = os.path.join(self.remote, self.split, basename)
        local = os.path.join(self.local, self.split, basename)
        download_or_wait(remote, local, wait, self.retry, self.timeout)
        return local

    def _download_shard(self, shard: int, partition: Partition) -> int:
        """Download the given shard.

        Args:
            shard (int): Shard ID.
            partition (Partition): Our rank and worker's partition of the dataset.

        Returns:
            int: Shard ID.
        """
        assert shard in partition.shards
        info = self.shards[shard]
        for raw_info, zip_info in info.file_pairs:
            if zip_info:
                raw_filename = os.path.join(self.local, self.split, raw_info.basename)
                if not os.path.isfile(raw_filename):
                    zip_filename = os.path.join(self.local, self.split, zip_info.basename)
                    if not os.path.isfile(zip_filename):
                        wait = shard not in partition.shards_to_download
                        self._download_file(zip_info.basename, wait)
                    data = open(zip_filename, 'rb').read()
                    if self.hash:
                        assert get_hash(self.hash, data) == zip_info.hashes[self.hash]
                    data = decompress(info.compression, data)  # pyright: ignore
                    with open(raw_filename, 'wb') as out:
                        out.write(data)
                    # if not self.keep_zip:
                    #     os.remove(zip_filename)
            else:
                raw_filename = os.path.join(self.local, self.split, raw_info.basename)
                if not os.path.isfile(raw_filename):
                    wait = shard not in partition.shards_to_download
                    self._download_file(raw_info.basename, wait)
                    if self.hash:
                        data = open(raw_filename, 'rb').read()
                        assert get_hash(self.hash, data) == raw_info.hashes[self.hash]
        return shard

    def _download_shards_via_pool(self,
                                  shards: List[int],
                                  partition: Partition,
                                  num_processes: Optional[int] = None) -> None:
        """Download and load the given missing shards.

        This is done in the main thread using a process pool.

        Args:
            shards (List[int]): The missing shards to download.
            partition (Partition): Our rank and worker's partition of the dataset.
            num_processes (Optional[int], default None): Number of concurrent shard downloads (ie,
                size of the process pool). If None, uses number of CPUs.
        """
        pool = Pool(num_processes)
        download_shard = lambda shard: self._download_shard(shard, partition)
        for shard in pool.imap_unordered(download_shard, shards):
            self._load_shard(shard, partition)
        with self._lock:
            self._download_status = DownloadStatus.DONE

    def _get_num_todo_samples(self) -> int:
        """Get the number of available samples.

        Returns:
            int: Number of available samples.
        """
        min_size = None
        with self._lock:
            for todo_ids in self._epoch_to_todo_ids.values():
                size = len(todo_ids)
                if min_size is None or size < min_size:
                    min_size = size
        return min_size or 0

    def _wait_until_few_todo_samples(self):
        """Block until the samples are low enough to download another shard."""
        if self.prefetch is None:
            return
        while True:
            if self._get_num_todo_samples() <= self.prefetch:
                break
            else:
                sleep(0.25)

    def _download_shards_via_loop(self, missing_shards: List[int], partition: Partition) -> None:
        """Sequentially download and load the given missing shards.

        This method is run in a background thread, which cannot use process pools because daemonic
        threads can't have child processes. In any case, with every worker downloading shards at
        once, process pool isn't necessary.

        Args:
            missing_shards (List[int]): The missing shards to download.
            partition (Partition): This rank and worker's part of the dataset.
        """
        for shard in missing_shards:
            self._wait_until_few_todo_samples()
            try:
                self._download_shard(shard, partition)
                self._load_shard(shard, partition)
            except Exception as e:
                with self._lock:
                    self._download_status = DownloadStatus.FAILED
                    self._download_exception = e
                return
        with self._lock:
            self._download_status = DownloadStatus.DONE

    def download(self, num_processes: Optional[int] = None) -> None:
        """Load all shards, downloading if not local (blocking).

        Args:
            num_processes (Optional[int], default None): Number of concurrent shard downloads (ie,
                size of the process pool). If None, uses number of CPUs.
        """
        partition = self.index.get_partition()
        shards = self._preload(partition)
        if shards:
            self._download_shards_via_pool(shards, partition, num_processes)

    def _start_downloading(self) -> bool:
        """Load shards in a thread, returning whether done immediately.

        Returns:
            bool: Whether all the shards were already local (are now loaded).
        """
        partition = self.index.get_partition()
        missing_shards = self._preload(partition)
        if missing_shards:
            Thread(target=self._download_shards_via_loop,
                   args=(missing_shards, partition),
                   daemon=True).start()
        with self._lock:
            return self._download_status == DownloadStatus.DONE

    def _iter_ids_static(self) -> Iterator[int]:
        """Get an iterator over all our sample IDs.

        Returns:
            Iterator[int]: Each sample ID.
        """
        ids = list(self._downloaded_ids)
        if self.shuffle:
            np.random.shuffle(ids)
        yield from ids

    def _iter_ids_dynamic(self) -> Iterator[int]:
        """Get an iterator over all our sample IDs as they become downloaded.

        If we are currently out of samples but not finished downloading the shards, blocks until it
        has new samples.

        Returns:
            Iterator[int]: Each sample ID.
        """
        with self._lock:
            epoch = self._next_epoch
            self._next_epoch += 1
            self._epoch_to_todo_ids[epoch] = todo_ids = list(self._downloaded_ids)
        while True:
            with self._lock:
                if self._download_status == DownloadStatus.IN_PROGRESS:
                    if todo_ids:
                        yield todo_ids.pop()
                        continue
                elif self._download_status == DownloadStatus.DONE:
                    if todo_ids:
                        yield todo_ids.pop()
                        continue
                    else:
                        del self._epoch_to_todo_ids[epoch]
                        break
                elif self._download_status == DownloadStatus.FAILED:
                    raise self._download_exception
                else:
                    raise RuntimeError('Unexpected download status.')
            sleep(0.25)

    def _iter_ids(self) -> Iterator[int]:
        """Get an iterator over all our sample IDs.

        Returns:
            Iterator[int]: Each sample ID.
        """
        if self._start_downloading():
            yield from self._iter_ids_static()
        else:
            yield from self._iter_ids_dynamic()

    def __getitem__(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        # Create lock in __getitem__ because we are prevented from putting it in __init__ because
        # of DataLoader num_workers and fork/spawn semantics.
        if not hasattr(self, '_lock'):
            self._lock = RLock()

        # Locate the shard and sample offset within that shard where the sample lives.
        shard_idx, idx_in_shard = self.index.find_sample(idx)

        # Load its shard if not loaded.
        with self._lock:
            if not self._has_shard[shard_idx]:
                partition = self.index.get_partition()
                self._download_shard(shard_idx, partition)
                self._load_shard(shard_idx, partition)

        # Now that we have the shard, load the sample there.
        shard = self.shards[shard_idx]
        return shard[idx_in_shard]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all the samples in our partition.

        If not all samples have been downloaded yet, iterates over what it has while inserting the
        remainder into the sequence behind the scenes as it progresses.

        Returns:
            Iterator[Any]: Each sample.
        """
        for idx in self._iter_ids():
            yield self[idx]
