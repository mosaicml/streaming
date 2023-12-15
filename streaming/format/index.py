# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Methods having to do with streaming dataset indexes."""

import json
import os
from re import Pattern
from typing import Callable, Dict, Iterable, Optional, Union
from warnings import warn

from streaming.format.parquet.indexing import index_parquet
from streaming.storage import CloudUploader, download_file, file_exists
from streaming.util.shorthand import normalize_duration

__all__ = ['get_index_basename']


def get_index_basename() -> str:
    """Get the canonical index file basename.

    Returns:
        str: Index basename.
    """
    return 'index.json'


Predicate = Union[str, Pattern, Callable[[str], bool]]


def materialize_index(*,
                      local: str,
                      remote: Optional[str] = None,
                      split: Optional[str] = None,
                      backend: str = 'streaming',
                      files: Optional[Iterable[str]] = None,
                      keep: Optional[Predicate] = r'^.*\.parquet$',
                      num_procs: Optional[int] = None,
                      show_progress: bool = True,
                      columns: Optional[Dict[str, Dict[str, str]]] = None,
                      match_columns: bool = True,
                      download_timeout: Union[float, str] = '5m',
                      max_file_size: Optional[Union[int, str]] = '200mb',
                      save_index_to_remote: bool = True) -> None:
    r"""Either download or generate the Streaming index for the given dataset.

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
            ``^.*\.parquet$``, i.e. include every file that ends with ".parquet".
        num_procs (int, optional): Number of processes for download/processing of potentially many
            large Parquet files. ``0`` means single-process; ``None`` means <number of CPUs>
            processes; positive int means that number of processes. Defaults to ``None``.
        show_progress (bool): Show progress bar for download/processing. Defaults to ``True``.
        columns (Dict[str, str], optional): For field names and types specified here, override the
            inferred columns to configure it manually. Defaults to ``None``.
        match_columns (bool): Whether to require that all the dataset Parquets have exactly the same
            column configuration. This is a correctness guard rail, preventing non-dataset Parquet
            shards from sneaking into our dataset. Streaming for its part is fine with shards being
            "incompatible"; assumes client will handle it. Defaults to ``True``.
        download_timeout (Union[float, str]): For each Parquet file. Defaults to ``2m``.
        max_file_size (Union[int, str], optional): File size limit, above which we raise an error.
            This is a performance guard rail, as choppiness increases linearly with shard size. The
            sweet spot is typically around 32mb. Defaults to ``200mb``.
        save_index_to_remote (bool): If we are indexing a third-party dataset and have a remote,
            whether to save the generated index to the remote in order to prevent having to index
            the dataset again in the future.
    """
    index_rel_path = get_index_basename()
    if backend == 'streaming':
        # First option: this is explicitly a Streaming dataset.
        #
        # Ensure the index.json is local and we're done.
        local_filename = os.path.join(local, split or '', index_rel_path)
        if not os.path.exists(local_filename):
            if remote:
                # Download the `index.json` to `index.json.tmp` then rename to `index.json`. This
                # is because only one process performs the downloading, while otherse wait for it
                # to complete.
                remote_path = os.path.join(remote, split or '', index_rel_path)
                temp_local_filename = local_filename + '.tmp'
                norm_download_timeout = normalize_duration(download_timeout)
                download_file(remote_path, temp_local_filename, norm_download_timeout)
                os.rename(temp_local_filename, local_filename)
            else:
                raise ValueError(f'No `remote` provided, but local file {local_filename} does ' +
                                 f'not exist either.')
    elif file_exists(local=local, remote=remote, split=split, path=index_rel_path):
        # Second option: this is a Streaming dataset, but the backend is set wrong.
        #
        # Note: Streaming datasets are datasets that Streaming can use -- they need a Streaming
        # index.json, but their shards can be in other formats, e.g. Parquet files or Delta tables.
        warn(f'Specified a non-Streaming backend ({backend}), but a Streaming index.json was ' +
             f'found (which makes this technically a Streaming dataset). Will use this ' +
             f'already-existing Streaming index instead of re-indexing the dataset.')
    else:
        # Third option: This is not a Streaming dataset.
        #
        # We call out to backend-specific assimilate() methods to index this third-party dataset,
        # resulting in a perfectly normal and valid index.json. May want to save that to remote.
        if backend == 'parquet':
            obj = index_parquet(local=local,
                                remote=remote,
                                split=split,
                                files=files,
                                keep=keep,
                                num_procs=num_procs,
                                show_progress=show_progress,
                                columns=columns,
                                match_columns=match_columns,
                                download_timeout=download_timeout,
                                max_file_size=max_file_size)
        else:
            raise ValueError(f'Unsupported backend: {backend}.')

        # Save index to local.
        index_filename = os.path.join(local, split or '', index_rel_path)
        with open(index_filename, 'w') as out:
            json.dump(obj, out)

        # Maybe save index to remote.
        if save_index_to_remote and remote:
            uploader = CloudUploader.get((local, remote), True)
            uploader.upload_file(index_rel_path)
