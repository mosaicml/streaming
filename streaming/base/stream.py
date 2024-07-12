# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A dataset, or sub-dataset if mixing, from which we stream/cache samples."""

import hashlib
import json
import os
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from streaming.base.compression import decompress
from streaming.base.constant import TICK
from streaming.base.distributed import barrier, get_local_rank
from streaming.base.format import FileInfo, Reader, get_index_basename, reader_from_json
from streaming.base.hashing import get_hash
from streaming.base.storage import download_file
from streaming.base.util import retry, wait_for_file_to_exist
from streaming.base.world import World


class Stream:
    """A dataset, or sub-dataset if mixing, from which we stream/cache samples.

    We initialize a StreamingDataset with one or more Streams. Streams may be resampled to achieve
    different mixtures of samples.

    Stream init takes three kinds of arguments:

    * At least one of ``remote`` and ``local`` must exist. If no ``remote``, the data must be
      local. If no ``local``, we cache to a temp directory.

      * ``remote``
      * ``local``

    * At most one of ``proportion``, ``repeat``, or ``choose`` may exist. If provided one of these,
      we derive the rest. Note that ``proportion`` (relative) and ``repeat``/``choose`` (absolute)
      are mutually incompatible -- you must entirely use one or the other (or neither) for all
      sub-datasets. If none are provided for all streams and ``epoch_size`` is unspecified, then
      each sample from each stream is seen once per epoch. If none are provided for all streams
      and ``epoch_size`` is specified, then streams are sampled in proportion to their size.

      * ``proportion``
      * ``repeat``
      * ``choose``

    * The remaining arguments are optional knobs for controlling downloading behavior and default
      to ``None``. If ``None``, they take a default value provided to or by the StreamingDataset
      init.

      * ``split``
      * ``download_retry``
      * ``download_timeout``
      * ``validate_hash``
      * ``keep_zip``

    Args:
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set. Defaults to
            ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        proportion (float, optional): How much to upsample or downsample this sub-dataset, as the
            proportion of the total combined dataset that consists of this sub-dataset. If
            using proportions, all sub-datasets provided together to the StreamingDataset init must
            define their proportions. The total combined number of samples is either the
            StreamingDataset argument "epoch_size" if provided, or kept the same total size as the
            underlying data if not. If provided, must be non-negative. Defaults to ``None``.
        repeat (float, optional): How much to upsample or downsample this sub-dataset, as a
            multipler on the number of samples. If provided, must be non-negative. Defaults to
            ``None``.
        choose (int, optional): How much to upsample or downsample this sub-dataset, as the exact
            number of resulting samples. If provided, must be non-negative. Defaults to ``None``.
        download_retry (int, optional): Number of download re-attempts before giving up. Defaults
            to ``None``.
        download_timeout (float, optional): Number of seconds to wait for a shard to download
            before raising an exception. Defaults to ``None``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool, optional): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep if and only if remote is local or no remote.
            Defaults to ``None``.
    """

    def __init__(self,
                 *,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 proportion: Optional[float] = None,
                 repeat: Optional[float] = None,
                 choose: Optional[int] = None,
                 download_retry: Optional[int] = None,
                 download_timeout: Optional[float] = None,
                 validate_hash: Optional[str] = None,
                 keep_zip: Optional[bool] = None) -> None:
        self.remote = remote
        self._local = local
        self.split = split or ''

        has_proportion = proportion is not None
        has_repeat = repeat is not None
        has_choose = choose is not None
        if not (0 <= has_proportion + has_repeat + has_choose <= 1):
            raise ValueError('At most one of `proportion`, `repeat`, and `choose` may be ' +
                             'specified; the others are derived')

        self._proportion = proportion
        if proportion is not None:
            if proportion < 0:
                raise ValueError('`proportion` must be non-negative')
            self.proportion = proportion

        self._repeat = repeat
        if repeat is not None:
            if repeat < 0:
                raise ValueError('`repeat` must be non-negative')
            self.repeat = repeat

        self._choose = choose
        if choose is not None:
            if choose < 0:
                raise ValueError('`choose` must be non-negative')
            self.choose = choose

        self._download_retry = download_retry
        if download_retry is not None:
            if download_retry < 0:
                raise ValueError('`download_retry` must be non-negative')
            self.download_retry = download_retry

        self._download_timeout = download_timeout
        if download_timeout is not None:
            if download_timeout <= 0:
                raise ValueError('`download_timeout` must be positive')
            self.download_timeout = download_timeout

        self.validate_hash = validate_hash

        if local is None:
            self.local = self._get_temporary_directory()
            if get_local_rank() == 0:
                if os.path.exists(self.local):
                    raise FileExistsError(
                        f'Could not create a temporary local directory {self.local} because it ' +
                        f'already exists. If you want to reuse the locally cached dataset, ' +
                        f'explicitly pass in a unique local directory with the `local` argument.' +
                        f' Otherwise, delete this directory and retry.')
                os.makedirs(self.local)
            barrier()
        else:
            self.local = local

        self._keep_zip = keep_zip
        if keep_zip is not None:
            self.keep_zip = keep_zip
            self.safe_keep_zip = self.keep_zip or self.remote in {None, self.local}

    def _get_temporary_directory(self) -> str:
        """Construct a path to a temporary directory based on remote and split."""
        root = tempfile.gettempdir()
        hash = ''
        if self.remote is not None:
            hash = hashlib.blake2s(self.remote.encode('utf-8'), digest_size=16).hexdigest()
        return os.path.join(root, hash, self.split)

    def apply_default(self, default: dict) -> None:
        """Apply defaults, setting any unset fields.

        We use pairs of (name, _name) in order to make type checking happy.

        Args:
            default (Self): Stream containing default values for all optional fields.
        """
        if not (self.remote or self._local):
            raise ValueError('`remote` and/or `local` path must be provided')

        if not self.split:
            self.split = default['split'] or ''
        if self._download_retry is None:
            self.download_retry = default['download_retry']
        if self._download_timeout is None:
            self.download_timeout = default['download_timeout']
        if self.validate_hash is None:
            self.validate_hash = default['validate_hash'] or None
        if self._keep_zip is None:
            self.keep_zip = default['keep_zip']
            self.safe_keep_zip = default['keep_zip'] or self.remote in {None, self.local}

    @classmethod
    def validate_weights(cls, streams: Sequence[Self]) -> Tuple[bool, bool]:
        """Validate stream weights, returning whether relative or absolute weighting was used.

        Args:
            streams (Sequence[Stream]): Every stream comprising the dataset.

        Returns:
            bool: Whether streams are weighted relatively (proportionally).
        """
        # Validate stream weights ("proportion", "repeat", "choose", or none).
        is_proportional = hasattr(streams[0], 'proportion')
        is_unspecified = True
        for stream_id, stream in enumerate(streams):
            has_proportion = hasattr(stream, 'proportion')
            has_repeat = hasattr(stream, 'repeat')
            has_choose = hasattr(stream, 'choose')
            if not (0 <= has_proportion + has_repeat + has_choose <= 1):
                raise ValueError(f'Streams must provide at most one of `proportion`, `repeat`, ' +
                                 f'or `choose` (error in stream {stream_id})')
            if is_proportional != has_proportion:
                raise ValueError(f'Relative (`proportion`) and absolute (`repeat`, `choose`, ' +
                                 f'none) stream weights are incompatible with each other (error ' +
                                 f'in stream {stream_id})')
            if has_proportion or has_repeat or has_choose:
                is_unspecified = False
        return is_proportional, is_unspecified

    @classmethod
    def apply_weights(cls, streams: Sequence[Self], samples_per_stream: NDArray[np.int64],
                      choose_per_epoch: Optional[int], seed: int) -> int:
        """Given samples per stream, derive each stream's proportion/repeat/samples.

        Modifies streams to save the derived weights.

        Args:
            streams (Sequence[Stream]): The list of streams which comprise the dataset.
            samples_per_stream (NDArray[np.int64]): Underlying samples of each stream.
            choose_per_epoch (int, optional): Absolute epoch size if weighting relatively.
            seed (int): Random number generator seed used to sample evenly.

        Returns:
            int: Number of samples to draw per epoch.
        """
        # Validate provided weights, determining whether they are relative or absolute.
        are_weights_relative, are_weights_unspecified = cls.validate_weights(streams)

        # Derive weights.
        if are_weights_relative:
            # Relative.
            if not choose_per_epoch:
                choose_per_epoch = sum(samples_per_stream)
            proportion_per_stream = np.array([stream.proportion for stream in streams], np.float64)
            proportion_per_stream /= proportion_per_stream.sum()
            choose_per_stream = (choose_per_epoch * proportion_per_stream).astype(np.int64)
            shortfall = choose_per_epoch - choose_per_stream.sum()
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(streams), shortfall, False)
            choose_per_stream[indices] += 1
            repeat_per_stream = choose_per_stream / samples_per_stream
        elif are_weights_unspecified and choose_per_epoch:
            # weights are unspecified, but epoch size (choose_per_epoch) is provided.
            # sample from each stream in proportion stream's samples
            proportion_per_stream = samples_per_stream.copy().astype(np.float64)
            proportion_per_stream /= proportion_per_stream.sum()
            choose_per_stream = (choose_per_epoch * proportion_per_stream).astype(np.int64)
            shortfall = choose_per_epoch - choose_per_stream.sum()
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(streams), shortfall, False)
            choose_per_stream[indices] += 1
            repeat_per_stream = choose_per_stream / samples_per_stream
        else:
            # Absolute.
            if choose_per_epoch:
                raise ValueError('Only provide `choose` when weighting streams relatively')
            choose_per_stream = np.zeros(len(streams), np.int64)
            for stream_id, stream in enumerate(streams):
                if hasattr(stream, 'repeat'):
                    choose = int(stream.repeat * samples_per_stream[stream_id])
                elif hasattr(stream, 'choose'):
                    choose = stream.choose
                else:
                    choose = samples_per_stream[stream_id]
                choose_per_stream[stream_id] = choose
            repeat_per_stream = choose_per_stream / samples_per_stream
            proportion_per_stream = choose_per_stream / choose_per_stream.sum()
            choose_per_epoch = sum(choose_per_stream)

        # Now that we know the true props/reps/choices, inject those back into the streams.
        for stream, proportion, repeat, choose in zip(streams, proportion_per_stream,
                                                      repeat_per_stream, choose_per_stream):
            stream.proportion = proportion
            stream.repeat = repeat
            stream.choose = choose

        return int(choose_per_epoch)

    def _download_file(self, from_basename: str, to_basename: Optional[str] = None) -> str:
        """Safely download a file from remote to local cache.

        Args:
            from_basename (str): Source basename.
            to_basename (str, optional): Destination basename, if different.

        Returns:
            str: Local cache filename.
        """
        # Calculate paths.
        if self.remote is None:
            remote = None
        else:
            remote = os.path.join(self.remote, self.split, from_basename)
        local = os.path.join(self.local, self.split, to_basename or from_basename)

        # Attempt to download, possibly repeating on failure.
        retry(num_attempts=self.download_retry)(
            lambda: download_file(remote, local, self.download_timeout))()

        return local

    def _decompress_shard_part(self, zip_info: FileInfo, zip_filename: str, raw_filename: str,
                               compression: Optional[str]) -> None:
        """Validate and decompress shard data.

        Args:
            zip_info (FileInfo): Compressed file info.
            zip_filename (str): Compressed filename.
            raw_filename (str): Decompressed filename.
            compression (str, optional): Compression algorithm.
        """
        # Load compressed.
        data = open(zip_filename, 'rb').read()

        # Validate what was downloaded.
        if self.validate_hash:
            if self.validate_hash not in zip_info.hashes:
                raise ValueError(
                    f'Hash algorithm `{self.validate_hash}` chosen for data ' +
                    f'validation does not match with those provided during dataset ' +
                    f'creation `{sorted(zip_info.hashes.keys())}`. Provide one of those.')
            if get_hash(self.validate_hash, data) != zip_info.hashes[self.validate_hash]:
                raise ValueError(f'Checksum failure: {zip_filename}')

        # Decompress and save that.
        data = decompress(compression, data)  # pyright: ignore
        tmp_filename = raw_filename + '.tmp'
        with open(tmp_filename, 'wb') as out:
            out.write(data)
        os.rename(tmp_filename, raw_filename)

        # Maybe remove compressed to save space.
        if not self.safe_keep_zip:
            os.remove(zip_filename)

    def _prepare_shard_part(self,
                            raw_info: FileInfo,
                            zip_info: Optional[FileInfo] = None,
                            compression: Optional[str] = None) -> int:
        """Get shard data given metadata for the raw and compressed versions of it.

        MDS format uses joint shards (ie, one file per shard). Other formats supported by streaming
        use split shards (ie, shard data lives in two files per shard: the raw data itself and
        metadata in a separate file).

        Args:
            raw_info (FileInfo): Raw file info.
            zip_info (FileInfo, optional): Zip file info. Defaults to ``None``.
            compression (str, optional): Compression algorithm used for zip_info. Defaults to
                ``None``.

        Returns:
            int: Change in cache usage.
        """
        # Has raw?
        delta = 0
        raw_filename = os.path.join(self.local, self.split, raw_info.basename)
        if os.path.isfile(raw_filename):
            # Has raw.
            if zip_info and not self.safe_keep_zip:
                zip_filename = os.path.join(self.local, self.split, zip_info.basename)
                if os.path.isfile(zip_filename):
                    # If don't keep zip and it has a zip, drop the zip.
                    os.remove(zip_filename)
                    delta -= zip_info.bytes
        else:
            # Missing raw. Uses zip?
            if zip_info:
                # Ensure has zip.
                zip_filename = os.path.join(self.local, self.split, zip_info.basename)
                if not os.path.isfile(zip_filename):
                    self._download_file(zip_info.basename)
                    delta += zip_info.bytes

                # Validate and decompress.
                self._decompress_shard_part(zip_info, zip_filename, raw_filename, compression)
                delta += raw_info.bytes
                if not self.safe_keep_zip:
                    delta -= zip_info.bytes
            else:
                # Download raw.
                self._download_file(raw_info.basename)
                delta += raw_info.bytes

                # Validate.
                if self.validate_hash:
                    if self.validate_hash not in raw_info.hashes:
                        raise ValueError(
                            f'Hash algorithm `{self.validate_hash}` chosen for data ' +
                            f'validation does not match with those provided during dataset ' +
                            f'creation `{sorted(raw_info.hashes.keys())}`. Provide one of those.')
                    data = open(raw_filename, 'rb').read()
                    if get_hash(self.validate_hash, data) != raw_info.hashes[self.validate_hash]:
                        raise ValueError(f'Checksum failure: {raw_filename}')
        return delta

    def prepare_shard(self, shard: Reader) -> int:
        """Ensure (download, validate, extract, etc.) that we have the given shard.

        Args:
            shard (Reader): Which shard.

        Returns:
            int: Change in cache usage.
        """
        delta = 0
        for raw_info, zip_info in shard.file_pairs:
            delta += self._prepare_shard_part(raw_info, zip_info, shard.compression)
        return delta

    def get_shards(self, world: World, allow_unsafe_types: bool) -> List[Reader]:
        """Load this Stream's index, retrieving its shard readers.

        Args:
            world (World): Distributed context.
            allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
                execution during deserialization, whether to keep going if ``True`` or raise an
                error.

        Returns:
            `List[Reader]: Shard readers.
        """
        # Download the index file if it does not exist locally.
        basename = get_index_basename()
        filename = os.path.join(self.local, self.split, basename)  # pyright: ignore
        if not os.path.exists(filename):
            if world.is_local_leader:
                if self.remote:
                    # Downloads the `index.json` as `index.json.tmp` fully and then rename it to
                    # `index.json` since only one process downloads the `index.json` file while
                    # other processes wait for it to get downloaded. Hence, It avoids loading the
                    # in-progress downloading `index.json`.
                    tmp_filename = self._download_file(basename, basename + '.tmp')
                    os.rename(tmp_filename, filename)
                else:
                    if not os.path.exists(filename):
                        raise RuntimeError(f'No `remote` provided, but local file {filename} ' +
                                           'does not exist either')
            else:
                wait_for_file_to_exist(
                    filename, TICK, self.download_timeout,
                    f'Index file {os.path.join(self.remote or "", self.split or "", basename)} ' +
                    f'-> {filename} took too long to download or failed to download. Either increase the '
                    + f'`download_timeout` value or check the local rank 0 traceback.')

        # Load the index.
        try:
            obj = json.load(open(filename))
        except json.decoder.JSONDecodeError as error:
            error.args = (f'Index file at {filename} is empty or corrupted. ' + error.args[0],)
            raise error

        # Version check.
        if obj['version'] != 2:
            raise ValueError(f'Unsupported streaming data version: {obj["version"]}. ' +
                             f'Expected version 2.')

        # Initialize shard readers according to the loaded info.
        shards = []
        for info in obj['shards']:
            shard = reader_from_json(self.local, self.split, info)
            shard.validate(allow_unsafe_types)
            shards.append(shard)

        return shards

    def set_up_local(self, shards: List[Reader], cache_usage_per_shard: NDArray[np.int64]) -> None:
        """Bring a local directory into a consistent state, getting which shards are present.

        Args:
            shards (List[Reader]): List of this stream's shards.
            cache_usage_per_shard (NDArray[np.int64]): Cache usage per shard of this stream.
        """
        # List the cache directory (so that we hit the filesystem once).
        local_dirname = os.path.join(self.local, self.split)
        listing = set()
        for dirname, _, subfiles in os.walk(local_dirname):
            for subfile in subfiles:
                filename = os.path.join(dirname, subfile)
                listing.add(filename)

        # Determine which shards are present, making local dir consistent.
        for i, shard in enumerate(shards):
            cache_usage_per_shard[i] = shard.set_up_local(listing, self.safe_keep_zip)

    def get_index_size(self) -> int:
        """Get the size of the index file in bytes.

        Returns:
            int: Size in bytes.
        """
        filename = os.path.join(self.local, self.split, get_index_basename())
        return os.stat(filename).st_size
