from enum import IntEnum
from math import ceil
from multiprocessing import Pool
import numpy as np
import os
from PIL import Image
from threading import RLock, Thread
from time import sleep
from torch.utils.data import Dataset, get_worker_info, IterableDataset
from torchvision.transforms.functional import to_tensor
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..compression import decompress
from .. import distributed as dist
from ..download import download_or_wait
from ..hashing import get_hash
from .index import get_index_basename, MDSIndex


class DownloadStatus(IntEnum):
    """Download status."""
    NOT_STARTED = 1
    IN_PROGRESS = 2
    DONE = 3
    FAILED = 4


class Partition(object):
    """A worker's partition of the dataset.

    Args:
        shards (List[int]): The shards that this partition overlaps.
        shards_to_download (List[int]): The shards that this worker should download (subset of
            ``shards``).
        min_id (int): The lowest sample ID of this partition.
        max_id (int): The highest sample ID of this partition.
    """

    def __init__(
        self,
        shards: List[int],
        shards_to_download: List[int],
        min_sample_id: int,
        max_sample_id: int
    ):
        self.shards = shards
        self.shards_to_download = shards_to_download
        self.min_sample_id = min_sample_id
        self.max_sample_id = max_sample_id


class MDSDataset(IterableDataset):
    """A sharded, streamed, iterable dataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        remote (Optional[str], default: None): Download shards from this remote path or directory.
            If None, this rank and workers' partition of the dataset must all exist locally.
        split (Optional[str], default: None): Which dataset split to use, if any.
        shuffle (bool, default: True): Whether to shuffle the samples while iterating.
        prefetch (Optional[int], default: 100_000): Target number of samples remaining to prefetch
            while iterating.
        keep_zip (Optional[bool], default: None): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote == local.
        retry (int, default: 2): Number of download re-attempts before giving up.
        timeout (float, default: 60): Number of seconds to wait for a shard to download before
            raising an exception.
        shard_hashes (Optional[List[str]], default: None): List of hash or checksum algorithms to
            use to validate shards.
        batch_size (Optional[int], default: None): Hint the batch_size that will be used on each
            device's DataLoader.

    .. doctest::

        To write the dataset:
        >>> import numpy as np
        >>> from PIL import Image
        >>> from uuid import uuid4
        >>> from streaming.base.mds.writer import MDSCreator
        >>> dirname = 'dirname'
        >>> fields = {
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
        >>> with MDSCreator(dirname, fields, compression, hashes) as out:
        ...     for sample in samples:
        ...         out.write(sample)

        To read the dataset:
        >>> from streaming.base.mds.dataset import MDSDataset
        >>> dataset = MDSDataset('dirname')
        >>> for sample in dataset:
        ...     print(sample)

        To read the dataset (with all optional arguments):
        >>> from streaming.base.mds.dataset import MDSDataset
        >>> dataset = MDSDataset(local='dirname', remote=None, split=None, shuffle=True,
        ...                      prefetch=100_000, keep_zip=None, retry=2, timeout=60,
        ...                      shard_hashes=None, batch_size=None)
    """

    def __init__(
        self,
        local: str,
        remote: Optional[str] = None,
        split: Optional[str] = None,
        shuffle: bool = True,
        prefetch: Optional[int] = 100_000,
        keep_zip: Optional[bool] = None,
        retry: int = 2,
        timeout: float = 60,
        shard_hashes: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> None:
        split = split or ''
        keep_zip = (remote == local) if keep_zip is None else keep_zip
        shard_hashes = shard_hashes or []

        self.local = local
        self.remote = remote
        self.split = split
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.keep_zip = keep_zip
        self.retry = retry
        self.timeout = timeout
        self.shard_hashes = shard_hashes
        self.batch_size = batch_size

        basename = get_index_basename()
        wait = dist.get_local_rank() != 0
        filename = self._download_file(basename, wait)
        self.index = MDSIndex.load(open(filename))

        self.samples_per_shard = np.zeros(len(self.index.shards), np.int64)
        for i, shard in enumerate(self.index.shards):
            self.samples_per_shard[i] = shard.samples
        self.total_samples = self.samples_per_shard.sum()
        self.shard_offsets = np.concatenate([np.zeros(1, np.int64),
                                             self.samples_per_shard.cumsum()])

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock: RLock
        self._has_shard = np.zeros(len(self.index.shards), np.uint8)
        self._next_epoch = 0
        self._epoch_to_todo_ids = {}
        self._downloaded_ids = []
        self._download_status = DownloadStatus.NOT_STARTED
        self._download_exception: Exception

    def __len__(self) -> int:
        """Get the length as an IterableDataset (ie, divided by number of devices).

        Returns:
            int: Dataset length.
        """
        return ceil(self.total_samples / dist.get_world_size())

    def get_partition(self) -> Partition:
        """Get the shards and sample range of a given partition of the dataset.

        When ``batch_size`` is provided, worker indices will be constructed so that there is at
        most one incomplete batch at the end of each epoch. For example, if the DataLoader is
        reading over::

            samples: [0, 1, 2, 3, 4, 5, 6, 7]
            num_workers: 3
            batch_size: 2
            drop_last: True

        but ``batch_size`` is not hinted to the StreamingDataset ahead of time, then the samples
        will by default be assigned like::

            worker 0: [0, 1, 2]
            worker 1: [3, 4, 5]
            worker 2: [6, 7]

        and will be read as batches like (with samples [2] and [5] dropped as incomplete)::

            batch 0: [0, 1]
            batch 1: [3, 4]
            batch 2: [6, 7]

        The above is suboptimal because we could have dropped no samples. So when ``batch_size`` is
        provided as a hint, we assign samples like this::

            worker 0: [0, 1, 2, 3]
            worker 1: [4, 5]
            worker 2: [6, 7]

        which will be read as batches like::

            batch 0: [0, 1]
            batch 1: [4, 5]
            batch 2: [6, 7]
            batch 3: [2, 3]

        Returns:
            Partition: This worker's partition of the dataset.
        """
        global_device = dist.get_global_rank()
        global_num_devices = dist.get_world_size()
        node_device = dist.get_local_rank()
        node_num_devices = dist.get_local_world_size()

        worker_info = get_worker_info()
        if worker_info:
            device_worker = worker_info.id
            device_num_workers = worker_info.num_workers
        else:
            device_worker = 0
            device_num_workers = 1
        node_worker = node_device * device_num_workers + device_worker
        node_num_workers = node_num_devices * device_num_workers

        # Splits a range (start, start+total) into num_parts such that:
        # each part spans a continguous range [part_min_id, part_max_id]
        # each part_i starts immediately from where the previous part_[i-1] stopped
        # all parts have the same number of items,
        # except the first K parts may have exactly 1 more item
        def _get_min_max_size(start: int, total: int, part: int, num_parts: int):
            sizes = [ceil((total - p) / num_parts) for p in range(num_parts)]
            min_ids = np.cumsum([0] + sizes)
            part_min_id = start + min_ids[part]
            part_max_id = start + min_ids[part + 1] - 1
            part_size = sizes[part]
            return part_min_id, part_max_id, part_size

        device_min_id, _, device_samples = _get_min_max_size(
            0, self.total_samples, global_device, global_num_devices)

        # Some devices may have 1 fewer sample, so repeat some samples at boundaries
        expected_device_samples = ceil(self.total_samples / global_num_devices)
        if device_samples < expected_device_samples:
            if device_samples != expected_device_samples - 1:
                raise RuntimeError('Found device partition with incorrect # samples')
            device_min_id -= 1
            device_samples += 1

        if not self.batch_size:
            worker_min_id, worker_max_id, _ = _get_min_max_size(
                device_min_id, device_samples, device_worker, device_num_workers)
        else:
            device_batches = ceil(device_samples / self.batch_size)
            samples_missing = device_batches * self.batch_size - device_samples

            # Determine which batches this worker is responsible for
            worker_min_batch_id, worker_max_batch_id, _ = _get_min_max_size(
                0, device_batches, device_worker, device_num_workers)

            # The last device_worker to be read from will be the one with the incomplete batch.
            # This is done to match PyTorch DataLoader's round-robin scheduling of workers.
            # All device_workers must be careful to account for the missing samples offset by the
            # incomplete batch.
            incomplete_device_worker = \
                (device_batches + device_num_workers - 1) % device_num_workers
            min_id_offset = 0 if device_worker <= incomplete_device_worker else samples_missing
            max_id_offset = 0 if device_worker < incomplete_device_worker else samples_missing

            worker_min_id = device_min_id + worker_min_batch_id * self.batch_size - min_id_offset
            worker_max_id = \
                device_min_id + (worker_max_batch_id + 1) * self.batch_size - max_id_offset - 1

        min_shard, _ = self._find_sample(worker_min_id)
        max_shard, _ = self._find_sample(worker_max_id)
        shards = list(range(min_shard, max_shard + 1))

        # Ensure that each shard only gets downloaded by 1 worker, so there are no race conditions.
        # To do this, we skip downloading the last shard (likely overlapped with the next worker)
        # unless:
        # - you are the last worker on your node (no files shared across nodes so you have to
        #   download it again!)
        # - you are downloading the last sample of the shard (no overlap with next worker)
        max_shard_next, _ = self._find_sample(worker_max_id + 1)
        if ((node_worker + 1 == node_num_workers) or
            (worker_max_id + 1 < self.total_samples and max_shard_next != max_shard)):
            shards_to_download = shards
        else:
            shards_to_download = shards[:-1]
        return Partition(shards, shards_to_download, worker_min_id, worker_max_id)

    def _load_shards(self, shards: List[int], partition: Partition) -> None:
        """Load the samples belonging to our partition from the given locally cached shards.

        Every time you call __iter__ on this dataset, it registers the list of samples you have
        left, which will not be the full epoch if the dataset isn't finished loaded when you start
        training.

        Calls to this method during training modify the samples remaining on these iterations on the
        fly to insert these new samples and then re-sort, making the shuffle as perfect as was
        possible.

        This operation is heavy and takes the lock, so call this method with all available shards at
        once.

        Args:
            shards (List[int]): Shard IDs.
            partition (Partition): Our rank and worker's partition of the dataset.
        """
        # Get our partition and shards' sample ranges.
        new_ids = []
        for shard in shards:
            shard_min_id = max(self.shard_offsets[shard], partition.min_sample_id)
            shard_max_id = min(self.shard_offsets[shard + 1] - 1, partition.max_sample_id)
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
        """Load the samples belonging to our partition from the given locally cached shard.

        For performance reasons, prefer _load_shards() where possible.

        Args:
            shard (int): Shard ID.
            partition (Partition): Our rank and worker's partition of the dataset.
        """
        self._load_shards([shard], partition)

    def _preload(self, partition: Partition) -> List[int]:
        """Load any shards that are cached locally, returning the list of missing shards.

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
            info = self.index.shards[shard]
            raw_filename = os.path.join(self.local, self.split, info.raw.basename)
            if os.path.isfile(raw_filename):
                if self.shard_hashes:
                    data = open(raw_filename, 'rb').read()
                    for algo in self.shard_hashes:
                        assert get_hash(algo, data) == info.raw.hashes[algo]
                present_shards.append(shard)
            elif not info.zip:
                missing_shards.append(shard)
            else:
                zip_filename = os.path.join(self.local, self.split, info.zip.basename)
                if os.path.isfile(zip_filename):
                    data = open(zip_filename, 'rb').read()
                    for algo in self.shard_hashes:
                        assert get_hash(algo, data) == info.zip.hashes[algo]
                    data = decompress(self.index.compression, data)  # pyright: ignore
                    with open(raw_filename, 'wb') as out:
                        out.write(data)
                    if not self.keep_zip:
                        os.remove(zip_filename)
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
        info = self.index.shards[shard]
        if info.zip:
            raw_filename = os.path.join(self.local, self.split, info.raw.basename)
            if not os.path.isfile(raw_filename):
                zip_filename = os.path.join(self.local, self.split, info.zip.basename)
                if not os.path.isfile(zip_filename):
                    wait = shard not in partition.shards_to_download
                    self._download_file(info.zip.basename, wait)
                data = open(zip_filename, 'rb').read()
                for algo in self.shard_hashes:
                    assert get_hash(algo, data) == info.zip.hashes[algo]
                data = decompress(self.index.compression, data)  # pyright: ignore
                with open(raw_filename, 'wb') as out:
                    out.write(data)
                if not self.keep_zip:
                    os.remove(zip_filename)
        else:
            raw_filename = os.path.join(self.local, self.split, info.raw.basename)
            if not os.path.isfile(raw_filename):
                wait = shard not in partition.shards_to_download
                self._download_file(info.raw.basename, wait)
                if self.shard_hashes:
                    data = open(raw_filename, 'rb').read()
                    for algo in self.shard_hashes:
                        assert get_hash(algo, data) == info.raw.hashes[algo]
        return shard

    def _download_shards_via_pool(self, shards: List[int], partition: Partition,
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
        """Block until the available samples are low enough to download another shard."""
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
        """Load all shards, downloading if not local.

        Args:
            num_processes (Optional[int], default None): Number of concurrent shard downloads (ie,
                size of the process pool). If None, uses number of CPUs.
        """
        partition = self.get_partition()
        shards = self._preload(partition)
        if shards:
            self._download_shards_via_pool(shards, partition, num_processes)

    def _start_downloading(self) -> bool:
        """Start loading all shards, downloading in a thread, returning whether done immediately.

        Returns:
            bool: Whether all the shards were already local (are now loaded).
        """
        partition = self.get_partition()
        missing_shards = self._preload(partition)
        if missing_shards:
            Thread(target=self._download_shards_via_loop, args=(missing_shards, partition),
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

    def _find_sample(self, idx: int) -> Tuple[int, int]:
        """Get the shard and offset where a sample will be found.

        Args:
            idx (int): Global sample index.

        Returns:
            Tuple[int, int]: Shard and sample index within that shard.
        """
        low = 0
        high = len(self.shard_offsets) - 1
        while True:
            if low + 1 == high:
                if idx == self.shard_offsets[high]:
                    shard = high
                else:
                    shard = low
                break
            mid = (low + high) // 2
            div = self.shard_offsets[mid]
            if idx < div:
                high = mid
            elif div < idx:
                low = mid
            else:
                shard = mid
                break
        offset = idx - self.shard_offsets[shard]
        return shard, offset

    def _get_shard_sample(self, shard: int, idx: int) -> Dict[str, Any]:
        """Get sample by shard and index within that shard, assuming its shard is loaded.

        Args:
            shard (int): Shard index.
            idx (int): Sample index within that shard.

        Returns:
            Dict[str, Any]: Sample data.
        """
        filename = os.path.join(self.local, self.split, self.index.shards[shard].raw.basename)
        offset = (1 + idx) * 4
        with open(filename, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
            fp.seek(begin)
            data = fp.read(end - begin)
        return self.index.decode_sample(data)

    def _get_sample(self, idx: int) -> Any:
        """Get sample by global index, assuming its shard is loaded.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        shard, idx_in_shard = self._find_sample(idx)
        return self._get_shard_sample(shard, idx_in_shard)

    def __getitem__(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if not loaded.

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
        shard, idx_in_shard = self._find_sample(idx)

        # Load its shard if not loaded.
        with self._lock:
            if not self._has_shard[shard]:
                partition = self.get_partition()
                self._download_shard(shard, partition)
                self._load_shard(shard, partition)

        # Now that we have the shard, load the sample there.
        return self._get_shard_sample(shard, idx_in_shard)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all the samples in our partition.

        If not all samples have been downloaded yet, iterates over what it has while inserting the
        remainder into the sequence behind the scenes as it progresses.

        Returns:
            Iterator[Any]: Each sample.
        """
        for idx in self._iter_ids():
            yield self[idx]
