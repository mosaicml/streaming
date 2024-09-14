# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A dataset, or sub-dataset if mixing, from which we stream/cache samples."""

import hashlib
import json
import time
import os
import tempfile
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from streaming.base.compression import decompress
from streaming.base.constant import TICK
import torch.distributed as dist
from streaming.base.distributed import barrier, get_local_rank
from streaming.base.format import FileInfo, Reader, get_index_basename, reader_from_json
from streaming.base.hashing import get_hash
from streaming.base.storage import download_file
from streaming.base.util import retry, wait_for_file_to_exist, wait_for_json_to_exist
from streaming.base.world import World

import re
import random
import pyarrow as pa
import requests
from tempfile import TemporaryDirectory


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

def save_dict_to_file(directory, filename, dictionary):
    """Save a dictionary to a file in the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {file_path}")

def load_dict_from_file(directory, filename):
    """Load a dictionary from a file in the specified directory."""
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    with open(file_path, 'r') as file:
        dictionary = json.load(file)
    print(f"Dictionary loaded from {file_path}")
    return dictionary


class DeltaSCStream(Stream):

    def __init__(self,
                 cluster_id: str,
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
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         proportion=proportion,
                         repeat=repeat,
                         choose=choose,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip)

        self.url_to_basename= {}
        self.basename_to_url={}
        self.cluster_id = cluster_id

    def generate_unique_basename(self, url: str, index: int) -> str:
        """Generate a unique basename for the file path from the URL."""
        hash_object = hashlib.md5(url.encode())
        hex_dig = hash_object.hexdigest()
        basename = '.'.join(['shard', f'{index:05}', 'mds'])
        self.url_to_basename[url] = basename
        self.basename_to_url[basename] = url

        return basename

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
        # Prepare cloudfetch
        from databricks.connect import DatabricksSession
        from databricks.sdk import WorkspaceClient
        from streaming.base.converters import infer_dataframe_schema

        w = WorkspaceClient()

        sparkSession = DatabricksSession.builder.remote(
            host=w.config.host,
            token=w.config.token,
            cluster_id=self.cluster_id).getOrCreate()

        df = sparkSession.sql(self.remote)
        query = df._plan.to_proto(df._session.client)  # pyright: ignore
        schema, cloudfetch_results = df._session.client.experimental_to_cloudfetch(query, "arrow", compression=False)  # pyright: ignore

        # Local leader prepares the index file based on cloudfetch results
        basename = get_index_basename()
        filename = os.path.join(self.local, self.split, basename)

        self.columns = infer_dataframe_schema(df, None)

        column_names = []
        column_encodings = []
        column_sizes = []
        for k, v in self.columns.items():
            column_names.append(k)
            column_encodings.append(v)
            column_sizes.append(None)

        if world.is_local_leader:

            metadata = {
                "version": 2,
                "shards": []
            }

            for index, result in enumerate(cloudfetch_results):
                shard = {
                    "column_encodings": column_encodings,
                    "column_names": column_names,
                    "column_sizes": column_sizes,
                    "compression": None,
                    "format": "mds",
                    "hashes": ["sha1"],
                    "raw_data": {
                        "basename": self.generate_unique_basename(result.url, index),
                        "bytes": result.uncompressed_size,
                        "hashes": {}
                    },
                    "samples": result.row_count,
                    "size_limit": 67108864,
                    "version": 2,
                    "zip_data": None
                }
                metadata["shards"].append(shard)

            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=4)

        else:
            wait_for_file_to_exist(
                filename, TICK, self.download_timeout,
                f'Index file {os.path.join(self.remote or "", self.split or "", basename)} ' +
                f'-> {filename} took too long to download. Either increase the ' +
                f'`download_timeout` value or check the other traceback.')

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

        save_dict_to_file('./', 'basename_to_url.json', self.basename_to_url)
        return shards

    def _download_file(self, from_basename: str, to_basename: Optional[str] = None) -> str:
        """Safely download a file from remote to local cache.

        Args:
            from_basename (str): Source basename.
            to_basename (str, optional): Destination basename, if different.

        Returns:
            str: Local cache filename.
        """
        from streaming import MDSWriter

        def fetch_and_convert(cloud_fetch_url: str, local_shard_path: str):
            samples = pa.ipc.open_stream(requests.get(cloud_fetch_url).content).read_all().to_pylist()

            with TemporaryDirectory() as temp_dir:
                with MDSWriter(columns=self.columns, out=temp_dir, size_limit=None) as out:
                    for sample in samples:
                        out.write(sample)
                temp_mds_filename = os.path.join(temp_dir, 'shard.00000.mds')
                os.rename(temp_mds_filename, local_shard_path)

        cloud_fetch_url = self.basename_to_url[from_basename]
        local = os.path.join(self.local, self.split, from_basename)

        # Attempt to download, possibly repeating on failure.
        retry(num_attempts=self.download_retry)(
            lambda: fetch_and_convert(cloud_fetch_url, local))()

        print('download to local is done = ', local)
        return local


class DeltaDBSQLStream(Stream):

    def __init__(self,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 proportion: Optional[float] = None,
                 repeat: Optional[float] = None,
                 choose: Optional[int] = None,
                 download_retry: Optional[int] = None,
                 download_timeout: Optional[float] = None,
                 validate_hash: Optional[str] = None,
                 keep_zip: Optional[bool] = None,
                 **kwargs: Any) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         proportion=proportion,
                         repeat=repeat,
                         choose=choose,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip)

        warehouse_id = kwargs.get('warehouse_id', None)
        host = kwargs.get('host', os.environ['DATABRICKS_HOST']).lstrip('https://')
        token = kwargs.get('token', os.environ['DATABRICKS_TOKEN'])
        catalog = kwargs.get('catalog', None)
        schema = kwargs.get('schema', None)

        if any([not warehouse_id, not host, not token, not catalog, not schema]):
            raise TypeError(f"Need to specify warehouse_id, host, token catalog, schema, during initialization, but got {warehouse_id}, {host}, {token}, {catalog}, {schema}")

        self.base_url = f"https://{host}/api/2.0/sql/statements/"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.data = {
            "warehouse_id": warehouse_id,
            "format": "ARROW_STREAM",
            "disposition": "EXTERNAL_LINKS",
            "statement": remote,
            "wait_timeout": "5s", # cannot be less than 5 otherwise throws bad request error
            "parameters": [],
            # "byte_limit": 10000000000000,
        }

        # From dbsql dtyps (lower case) to MDS encoded types
        # https://docs.databricks.com/en/dev-tools/python-sql-connector.html
        self.dtypes_mapping = {
            'string' : 'str',
            'bigint' : 'int64',
            'array': 'ndarray',
            'array<string>': 'str_array',
            'binary': 'bytes',
            'boolean': 'uint32',
            'date': 'str',
            'datetime.date': 'str',
            'decimal': 'str_decimal',
            'double' : 'float64',
            'int': 'int',
            'map': 'json',
            'smallint': 'int16',
            'struct': 'json',
            'tinyint': 'int8',
            'long': 'int8',
            'array<struct<content: string, role: string>>': 'json', # special for messages
        }

    def generate_statement_id_and_sync(self, world: World):
        if dist.is_available() and dist.is_initialized():
            barrier()

            if world.is_leader: # is_local_leader:
                response = requests.post(self.base_url, headers=self.headers, json=self.data)
                response.raise_for_status()
                response_data = response.json()
                self.statement_id = response_data['statement_id']
                data = self.statement_id
            else:
                data = None


            obj_list = [data]
            dist.broadcast_object_list(obj_list, src=0)
            self.statement_id = obj_list[0]
            return

        world_size = world.num_ranks
        if world_size > 1:
            raise RuntimeError(''.join([
                f'The world_size({world_size}) > 1, but the distributed package is not available ',
                'or has not been initialized. Please check you have initialized the distributed ',
                'runtime and that PyTorch has been built with distributed support.'
            ]))

        response = requests.post(self.base_url, headers=self.headers, json=self.data)
        response.raise_for_status()
        response_data = response.json()
        self.statement_id = response_data['statement_id']

    def wait_for_query_result(self, timeout=3600):
        if not self.statement_id:
            raise ValueError(f"statement id is not set yet")

        total_time = 0
        while total_time <= timeout:
            response = requests.get(f"{self.base_url}/{self.statement_id}", headers=self.headers)
            response.raise_for_status()
            response_data = response.json()
            query_status = response_data['status']['state']

            if query_status == "SUCCEEDED":
                #self.statement_id = response_data['statement_id']
                save_dict_to_file(self.local, f'response_{int(time.time())}', response_data)
                return response_data

            print(f"Query status: {query_status}")
            time.sleep(3)
            total_time += 3
        raise TimeoutError(f"Query execution failed with status: {query_status}")

    def get_encode_format(self, sql_fmt: str):
        mds_fmt = self.dtypes_mapping.get(sql_fmt.lower(), None)
        if not mds_fmt:
            raise TypeError(f"{sql_fmt} is not supported by MDSWrite.")
        return mds_fmt

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
        from streaming.base.format.mds.encodings import (get_mds_encoded_size, get_mds_encodings,
                                                         is_mds_encoding, mds_encode)
        self.generate_statement_id_and_sync(world)

        sql_response = self.wait_for_query_result()

        # Local leader prepares the index file based on cloudfetch results
        basename = get_index_basename()
        filename = os.path.join(self.local, self.split, basename)

        self.columns = { c['name']:  self.get_encode_format(c['type_text']) for c in  sql_response['manifest']['schema']['columns'] }

        column_names = []
        column_encodings = []
        column_sizes = []
        for name in sorted(self.columns):
            encoding = self.columns[name]
            if not is_mds_encoding(encoding):
                raise TypeError(f'MDSWriter passed column `{name}` with encoding `{encoding}` ' +
                                f'is unsupported. Supported encodings are {get_mds_encodings()}')
            size = get_mds_encoded_size(encoding)
            column_names.append(name)
            column_encodings.append(encoding)
            column_sizes.append(size)

        print(f'self.columns = {self.columns}')

        total_shard_count = sql_response['manifest']['total_chunk_count']

        if world.is_local_leader:

            metadata = {
                "version": 2,
                "shards": []
            }

            for shard_id, shard_meta in enumerate(sql_response['manifest']['chunks']):
                shard = {
                    "column_encodings": column_encodings,
                    "column_names": column_names,
                    "column_sizes": column_sizes,
                    "compression": None,
                    "format": "mds",
                    "hashes": ["sha1"],
                    "raw_data": {
                        "basename": f'shard.{shard_id:05}.mds',
                        "bytes": shard_meta['byte_count'],
                        "hashes": {}
                    },
                    "samples": shard_meta['row_count'],
                    "size_limit": 67108864,
                    "version": 2,
                    "zip_data": None
                }
                metadata["shards"].append(shard)

            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=4)
        else:
            wait_for_json_to_exist(
                filename, TICK, self.download_timeout,
                f'Index file {os.path.join(self.remote or "", self.split or "", basename)} ' +
                f'-> {filename} took too long to download. Either increase the ' +
                f'`download_timeout` value or check the other traceback.')

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

    def _make_request(self, url: str) -> requests.Response:
        if random.random() < 0.0:  # make rhs > 0.0 for testing, so x% of the time return HTTPError
            response = requests.Response()
            response.status_code = 404
            response.url = url
            raise requests.exceptions.HTTPError(f"Manually raised HTTPError for testing purposes: {int(time.time())}", response=response)
        else:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response

    def _download_file(self, from_basename: str, to_basename: Optional[str] = None) -> str:
        """Safely download a file from remote to local cache.

        Args:
            from_basename (str): Source basename.
            to_basename (str, optional): Destination basename, if different.

        Returns:
            str: Local cache filename.
        """
        from streaming import MDSWriter
        def _fetch_and_convert(cloud_fetch_url: str, local_shard_path: str):
            samples = pa.ipc.open_stream(requests.get(cloud_fetch_url).content).read_all().to_pylist()
            with TemporaryDirectory() as temp_dir:
                with MDSWriter(columns=self.columns, out=temp_dir, size_limit=None) as out:
                    for sample in samples:
                        out.write(sample)
                temp_mds_filename = os.path.join(temp_dir, 'shard.00000.mds')
                os.rename(temp_mds_filename, local_shard_path)

        chunk_index = int(re.search(r'\d+', from_basename).group())
        print('from_basename = ', from_basename)
        print('chunk_index = ', chunk_index)


        try:
            url = f"{self.base_url}/{self.statement_id}/result/chunks/{chunk_index}"
            response = self._make_request(url)
        except Exception as e: # requests.exceptions.HTTPError as e:
            print('Failed to download, I cannot refresh statement id and try again')
            print('url = ', url)
            print(e)
            raise TimeoutError('Check if the query results retention period of your workspace and make sure it is longer than the expected training period. For multi-node, we do not want to refresh and communicate statement id from worker processes.') from e
            # self.refresh_statement_id()
            #url = f"{self.base_url}/{self.statement_id}/result/chunks/{chunk_index}"
            #response = self._make_request(url)

        cloud_fetch_url = response.json()['external_links'][0]['external_link']
        local = os.path.join(self.local, self.split, from_basename)
        retry(num_attempts=self.download_retry)(lambda: _fetch_and_convert(cloud_fetch_url, local))()

        print('Download to local is done = ', local)
        return local


class DeltaDBSQLStreamSession(Stream):

    def __init__(self,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 proportion: Optional[float] = None,
                 repeat: Optional[float] = None,
                 choose: Optional[int] = None,
                 download_retry: Optional[int] = None,
                 download_timeout: Optional[float] = None,
                 validate_hash: Optional[str] = None,
                 keep_zip: Optional[bool] = None,
                 **kwargs: Any) -> None:
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         proportion=proportion,
                         repeat=repeat,
                         choose=choose,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip)

        warehouse_id = kwargs.get('warehouse_id', None)
        host = kwargs.get('host', os.environ['DATABRICKS_HOST'])
        token = kwargs.get('token', os.environ['DATABRICKS_TOKEN'])
        catalog = kwargs.get('catalog', None)
        schema = kwargs.get('schema', None)
        self.use_cached_result = kwargs.get('use_cached_result', False)

        if any([not warehouse_id, not host, not token, not catalog, not schema]):
            raise TypeError(f"Need to specify warehouse_id, host, token catalog, schema, during initialization")

        self.base_url = f"https://{host}/api/2.0/sql/statements/"
        self.session_url = f"https://{host}/api/2.0/sql/sessions/"

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        self.session_payload = {
            "warehouse_id": warehouse_id,
            "catalog": catalog,
            "schema": schema,
            "session_confs": {"use_cached_result": "false"}
        }

        self.payload = {
            "warehouse_id": warehouse_id,
            "format": "ARROW_STREAM",
            "disposition": "EXTERNAL_LINKS",
            "statement": remote,
            "wait_timeout": "5s", # cannot be less than 5 otherwise throws bad request error
            "parameters": [],
        }

        # From dbsql dtyps (lower case) to MDS encoded types
        # https://docs.databricks.com/en/dev-tools/python-sql-connector.html
        self.dtypes_mapping = {
            'string' : 'str',
            'bigint' : 'int64',
            'array': 'ndarray',
            'array<string>': 'str_array',
            'binary': 'bytes',
            'boolean': 'uint32',
            'date': 'str',
            'datetime.date': 'str',
            'decimal': 'str_decimal',
            'double' : 'float64',
            'int': 'int',
            'map': 'json',
            'smallint': 'int16',
            'struct': 'json',
            'tinyint': 'int8',
            'long': 'int8',
        }

        self.refresh_statement_id(self.use_cached_result)

    def polling(self, timeout: int = 3600):
        total_time = 0
        while total_time <= timeout:
            response = requests.get(f"{self.base_url}/{self.statement_id}", headers=self.headers)
            response.raise_for_status()
            response_data = response.json()
            query_status = response_data['status']['state']

            if query_status == "SUCCEEDED":
                save_dict_to_file(self.local, f'response_{int(time.time())}', response_data)
                return response_data

            print(f"Query status: {query_status}")
            time.sleep(3)
            total_time += 3
        raise TimeoutError(f"Query execution failed with status: {query_status}")


    def refresh_statement_id(self, use_cached_result:bool=False):

        boolean_string = "true" if use_cached_result else "false"
        self.session_payload['session_confs']['use_cached_result'] = boolean_string

        print(f"Set the session data to be {self.session_payload}")

        # Create a session id
        # Use session id in payload
        # Fetch result via get status api
        response = requests.post(self.session_url, headers=self.headers, json=self.session_payload)
        self.payload['session_id'] = response.json()['session_id']

        print(f"Set the payload to be {self.payload}")

        response = requests.post(self.base_url, headers=self.headers, json=self.payload)
        response.raise_for_status()
        response_data = response.json()
        self.statement_id = response_data['statement_id']

        return self.polling()

    def get_encode_format(self, sql_fmt: str):
        mds_fmt = self.dtypes_mapping.get(sql_fmt.lower(), None)
        if not mds_fmt:
            raise TypeError(f"{sql_fmt} is not supported by MDSWrite.")
        return mds_fmt

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
        from streaming.base.format.mds.encodings import (get_mds_encoded_size, get_mds_encodings,
                                                         is_mds_encoding, mds_encode)

        sql_response = self.refresh_statement_id(True)

        # Local leader prepares the index file based on cloudfetch results
        basename = get_index_basename()
        filename = os.path.join(self.local, self.split, basename)

        self.columns = { c['name']:  self.get_encode_format(c['type_text']) for c in  sql_response['manifest']['schema']['columns'] }

        column_names = []
        column_encodings = []
        column_sizes = []
        for name in sorted(self.columns):
            encoding = self.columns[name]
            if not is_mds_encoding(encoding):
                raise TypeError(f'MDSWriter passed column `{name}` with encoding `{encoding}` ' +
                                f'is unsupported. Supported encodings are {get_mds_encodings()}')
            size = get_mds_encoded_size(encoding)
            column_names.append(name)
            column_encodings.append(encoding)
            column_sizes.append(size)

        print(f'self.columns = {self.columns}')

        total_shard_count = sql_response['manifest']['total_chunk_count']

        if world.is_local_leader:

            metadata = {
                "version": 2,
                "shards": []
            }

            for shard_id, shard_meta in enumerate(sql_response['manifest']['chunks']):
                shard = {
                    "column_encodings": column_encodings,
                    "column_names": column_names,
                    "column_sizes": column_sizes,
                    "compression": None,
                    "format": "mds",
                    "hashes": ["sha1"],
                    "raw_data": {
                        "basename": f'shard.{shard_id:05}.mds',
                        "bytes": shard_meta['byte_count'],
                        "hashes": {}
                    },
                    "samples": shard_meta['row_count'],
                    "size_limit": 67108864,
                    "version": 2,
                    "zip_data": None
                }
                metadata["shards"].append(shard)

            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=4)
        else:
            wait_for_json_to_exist(
                filename, TICK, self.download_timeout,
                f'Index file {os.path.join(self.remote or "", self.split or "", basename)} ' +
                f'-> {filename} took too long to download. Either increase the ' +
                f'`download_timeout` value or check the other traceback.')

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

    def _make_request(self, url: str) -> requests.Response:
        if random.random() < 0.0:  # make rhs > 0.0 for testing, so x% of the time return HTTPError
            response = requests.Response()
            response.status_code = 404
            response.url = url
            raise requests.exceptions.HTTPError(f"Manually raised HTTPError for testing purposes: {int(time.time())}", response=response)
        else:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response

    def _download_file(self, from_basename: str, to_basename: Optional[str] = None) -> str:
        """Safely download a file from remote to local cache.

        Args:
            from_basename (str): Source basename.
            to_basename (str, optional): Destination basename, if different.

        Returns:
            str: Local cache filename.
        """
        from streaming import MDSWriter
        def _fetch_and_convert(cloud_fetch_url: str, local_shard_path: str):
            samples = pa.ipc.open_stream(requests.get(cloud_fetch_url).content).read_all().to_pylist()
            with TemporaryDirectory() as temp_dir:
                with MDSWriter(columns=self.columns, out=temp_dir, size_limit=None) as out:
                    for sample in samples:
                        out.write(sample)
                temp_mds_filename = os.path.join(temp_dir, 'shard.00000.mds')
                os.rename(temp_mds_filename, local_shard_path)

        chunk_index = int(re.search(r'\d+', from_basename).group())
        print('from_basename = ', from_basename)
        print('chunk_index = ', chunk_index)

        try:
            url = f"{self.base_url}/{self.statement_id}/result/chunks/{chunk_index}"
            response = self._make_request(url)
        except Exception as e: # requests.exceptions.HTTPError as e:
            print('Failed to download, refresh statement id and try again')
            print('url = ', url)
            print(e)
            self.refresh_statement_id(True)
            url = f"{self.base_url}/{self.statement_id}/result/chunks/{chunk_index}"
            response = self._make_request(url)

        cloud_fetch_url = response.json()['external_links'][0]['external_link']
        local = os.path.join(self.local, self.split, from_basename)
        retry(num_attempts=self.download_retry)(lambda: _fetch_and_convert(cloud_fetch_url, local))()

        print('Download to local is done = ', local)
        return local

