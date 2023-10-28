# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Indexing a Parquet dataset for use by Streaming."""

from re import Pattern
from typing import Any, Callable, Dict, Iterable, Optional, Union

Filter = Union[str, Pattern, Callable[[str], bool]]


def index_parquet(*,
                  local: str,
                  remote: Optional[str] = None,
                  split: Optional[str] = None,
                  files: Optional[Iterable[str]] = None,
                  keep: Optional[Filter] = r'.*\.parquet$',
                  num_procs: Optional[int] = 0,
                  download_timeout: Union[float, str] = '2m',
                  max_file_bytes: Optional[Union[int, str]] = '200mb',
                  same_schema: bool = True,
                  columns: Optional[Dict[str, str]] = None,
                  show_progress: bool = True) -> Dict[str, Any]:
    r"""Initialize from a local and/or remote Parquet dataset directory.

    "Parquet dataset" means the samples live in a collection of naked Parquet files. There is not
    any kind of index or manifest we can count on existing, so we will have to create one.

    Assumptions:
      * Samples live in a collection of naked Parquet files.
      * There is not any kind of index or manifest that we can count on existing.
      * Files are all found under a common root directory, which local/remote point to.
      * This root directory may contain other files, which we ignore.
      * Ideally, but not necessarily, the Parquets all have the same schema.

    Locality:
      * If we are given an explicit list of Parquet files, we try local first, then remote. Both
        are cross-checked for completeness.
      * If we are default listing all files instead, and just have a local, it is assumed to be
        complete.
      * If we are listing files, and remote is provided, the remote must be authoritative.

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
            large Parquet files. ``None`` means single-process; ``0`` means <number of CPUs>
            processes; positive int means that number of processes. Defaults to ``0``.
        download_timeout (Union[float, str]): For each Parquet file. Defaults to ``2m``.
        max_file_bytes (Union[int, str], optional): File size limit, above which we raise an error.
            This is a performance guard rail, as choppiness increases linearly with shard size. The
            sweet spot is typically around 32mb. Defaults to ``200mb``.
        same_schema (bool): Whether to require that all the dataset Parquets have exactly the same
            Parquet schema. This is a correctness guard rail, preventing non-dataset Parquet shards
            from sneaking into our dataset. Streaming for its part is fine with shards being
            "incompatible"; assumes client will handle it. Defaults to ``True``.
        columns (Dict[str, str], optional): For field names and types specified here, override the
            inferred schema to configure it manually. Defaults to ``None``.
        show_progress (bool): Show progress bar for download/processing. Defaults to ``True``.

    Returns:
        Dict[str, Any]: StreamingDataset index configuration to stream this Parquet dataset.
    """
    raise NotImplementedError  # TODO
