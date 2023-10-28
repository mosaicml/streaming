# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Index a Delta table for use by Streaming."""

from typing import Any, Dict, Optional, Union

__all__ = ['index_delta']


def index_delta(*,
                local: str,
                remote: Optional[str] = None,
                split: Optional[str] = None,
                version: Optional[int] = None,
                num_threads: Optional[int] = 0,
                download_timeout: Union[float, str] = '1m',
                max_file_bytes: Optional[Union[int, str]] = '200mb',
                columns: Optional[Dict[str, Optional[str]]] = None,
                show_progress: bool = True) -> Dict[str, Any]:
    """Index a local and/or remote Delta table directory for use by Streaming.

    Args:
        local (str): Where the dataset is cached on the local filesystem.
        remote (str, optional): Where the dataset is downloaded from. Defaults to ``None``.
        split (str, optional): Which dataset split to use. Defaults to ``None``.
        version (int, optional): Which snapshot version of the dataset to use, or else take the
            latest if ``None``. Defaults to ``None``.
        num_threads (int, optional): Number of threads for downloading potentially many very small
            files. ``None`` means single-threaded; ``0`` means <number of CPUs> threads; positive
            int means that number of threads. Default: ``0``.
        download_timeout (Union[float, str]): For each Delta metadata file. Defaults to ``1m``.
        max_file_bytes (Union[int, str], optional): File size limit, above which we raise an error.
            This is a performance guard rail, as choppiness increases linearly with shard size. The
            sweet spot is typically around 32mb. Defaults to ``200mb``.
        columns (Dict[str, str], optional): For field names and types specified here, override the
            inferred schema to configure it manually. Defaults to ``None``.
        show_progress (bool): Show progress bar for downloading Delta logs. Defaults to ``True``.

    Returns:
        Dict[str, Any]: StreamingDataset index configuration to stream this Delta table.
    """
    raise NotImplementedError  # TODO
