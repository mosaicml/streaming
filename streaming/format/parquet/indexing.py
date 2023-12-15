# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Indexing a Parquet dataset for use by Streaming."""

import os
from re import Pattern
from typing import Any, Callable, Dict, Iterable, Optional, Union

from pyarrow import parquet as pq
from tqdm import tqdm

from streaming.format.mds.encodings import get_mds_encoded_size
from streaming.storage.extra import list_dataset_files, smart_download_file
from streaming.util.shorthand import normalize_duration

__all__ = ['index_parquet']


def _get_mds_column(val: Any) -> str:
    """Get the MDS column encoding of one field.

    Args:
        val (Any): The field.

    Returns:
        str: Its corresponding MDS encoding.
    """
    if isinstance(val, int):
        return 'int'
    elif isinstance(val, str):
        return 'str'
    else:
        raise ValueError('Unsupported column type: {type(val)}.')


def _sample_to_columns(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Get column names, encodings, and sizes.

    Args:
        sample (Dict[str, Any]): A sample to derive column info from.

    Returns:
        Dict[str, Any]: MDS column names, encodings, and sizes.
    """
    col_names = sorted(sample)
    col_encs = []
    for name in col_names:
        val = sample[name]
        enc = _get_mds_column(val)
        col_encs.append(enc)
    col_sizes = list(map(get_mds_encoded_size, col_encs))
    return {
        'column_names': col_names,
        'column_encodings': col_encs,
        'column_sizes': col_sizes,
    }


def _index_file(local: str,
                remote: Optional[str],
                split: Optional[str],
                rel_path: str,
                download_timeout: Union[float, str] = '2m',
                max_file_bytes: Optional[Union[int, str]] = '200mb',
                want_mds_columns: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get info a Streaming index needs about a Parquet shard.

    Args:
        local (str): Local dataset root.
        remote (str, optional): Remote dataset root, if remote is provided.
        split (str, optional): Split, if used.
        rel_path (str): Path to file, relative to serialized dataset root.
        download_timeout (Union[float, str]): Maximum download time. Defaults to ``2m``.
        max_file_bytes (Union[int, str], optional): Maximum file size. This is to catch people
            trying to stream gigantic Parquet shards. Defaults to ``200mb``.
        want_mds_columns (Dict[str, Any], optional): If provided, MDS schemna that this Parquet
            shard must match upon conversion to MDS.

    Returns:
        Dict[str, Any]: Shard info, or None upon failure.
    """
    local_path = os.path.join(local, split or '', rel_path)
    if not os.path.exists(local):
        if not remote:
            raise ValueError('Remote was needed, but not provided.')

        remote_path = os.path.join(remote, split or '', rel_path)
        smart_download_file(remote=remote_path,
                            local=local_path,
                            timeout=download_timeout,
                            max_size=max_file_bytes)

    num_bytes = os.stat(local).st_size

    table = pq.read_table(local_path)
    samples = table.to_pylist()
    num_samples = len(samples)
    mds_columns = _sample_to_columns(samples[0])
    if want_mds_columns and want_mds_columns != mds_columns:
        raise ValueError(f'MDS column mismatch: required {want_mds_columns}, but got ' +
                         f'{mds_columns}.')

    ret = {
        'version': 2,
        'format': 'parquet',
        'raw_parquet': {
            'basename': rel_path,
            'bytes': num_bytes,
        },
        'raw_data': {
            'basename': rel_path + '.mds',
        },
        'samples': num_samples,
    }
    ret.update(mds_columns)
    return ret


def _shard_metadata_to_columns(info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract MDS column information from the info for a shard.

    Args:
        info (Dict[str, Any]): Shard info.

    Returns:
        Dict[str, Any]: MDS columns.
    """
    ret = {}
    for key in ['column_names', 'column_encodings', 'column_sizes']:
        ret[key] = info[key]
    return ret


Predicate = Union[str, Pattern, Callable[[str], bool]]


def index_parquet(*,
                  local: str,
                  remote: Optional[str] = None,
                  split: Optional[str] = None,
                  files: Optional[Iterable[str]] = None,
                  keep: Optional[Predicate] = r'.*\.parquet$',
                  num_procs: Optional[int] = None,
                  show_progress: bool = True,
                  columns: Optional[Dict[str, Dict[str, str]]] = None,
                  match_columns: bool = True,
                  download_timeout: Union[float, str] = '5m',
                  max_file_size: Optional[Union[int, str]] = '200mb') -> Dict[str, Any]:
    r"""Index a local and/or remote Parquet dataset directory for use by Streaming.

    "Parquet dataset" means the samples live in a collection of naked Parquet files. There is not
    any kind of index or manifest we can count on existing, so we will have to create one.

    Assumptions:
      * Samples live in a collection of naked Parquet files.
      * There is not any kind of index or manifest that we can count on existing.
      * Files are all found under a common root directory, which local/remote point to.
      * This root directory may contain other files, which we ignore.
      * Ideally, but not necessarily, the Parquets all have the same columns.

    Locality:
      * If we are given an explicit list of Parquet files, we try local first, then remote. Both
        are cross-checked for completeness.
      * If we are default listing all files instead, and just have a local, it is assumed to be
        complete.
      * If we are listing files, and remote is provided, the remote must be authoritative.

    TODO: use num_procs.
    TODO: use columns.

    Args:
        local (str): Where the dataset is cached on the local filesystem.
        remote (str, optional): Where the dataset is downloaded from. Defaults to ``None``.
        split (str, optional): Which dataset split to use. Defaults to ``None``.
        files (Iterable[str], optional): An Iterable of file paths relative to dataset root. These
            paths filtered for the Parquets constituting this dataset by ``keep``. If not set, we
            default to a sorted listing of all the files under dataset root. We list the remote if
            provided, else we assume local is complete. Defaults to ``None``.
        keep (Union[str, Pattern, Callable[[str], bool]], optional): Iterating ``files``, we keep
            the ones this regex matches (if str) or predicate accepts (if Callable). Defaults to
            ``.*\.parquet$``, i.e. include every file that ends with ".parquet".
        num_procs (int, optional): Number of processes for download/processing of potentially many
            large Parquet files. ``0`` means single-process; ``None`` means <number of CPUs>
            processes; positive int means that number of processes. Defaults to ``None``.
        show_progress (bool): Show progress bar for download/processing. Defaults to ``True``.
        columns (Dict[str, Dict[str, str]], optional): For field names and types specified here,
            override the inferred columns to configure it manually. Defaults to ``None``.
        match_columns (bool): Whether to require that all the dataset Parquets have exactly the same
            Parquet columns. This is a correctness guard rail, preventing non-dataset Parquet shards
            from sneaking into our dataset. Streaming for its part is fine with shards being
            "incompatible"; assumes client will handle it. Defaults to ``True``.
        download_timeout (Union[float, str]): For each Parquet file. Defaults to ``2m``.
        max_file_size (Union[int, str], optional): File size limit, above which we raise an error.
            This is a performance guard rail, as choppiness increases linearly with shard size. The
            sweet spot is typically around 32mb. Defaults to ``200mb``.

    Returns:
        Dict[str, Any]: StreamingDataset index configuration to stream this Parquet dataset.
    """
    norm_download_timeout = normalize_duration(download_timeout)

    rel_paths = list_dataset_files(local=local, remote=remote, split=split, paths=files, keep=keep)
    if show_progress:
        rel_paths = tqdm(rel_paths, leave=False)

    want_mds_columns = None
    infos = []
    for rel_path in rel_paths:
        info = _index_file(local, remote, split, rel_path, norm_download_timeout, max_file_size,
                           want_mds_columns)
        infos.append(info)

        if match_columns and not want_mds_columns:
            want_mds_columns = _shard_metadata_to_columns(info)

    return {
        'version': 2,
        'shards': infos,
    }
