# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Indexing a Lance dataset for use by Streaming."""

from typing import Any, Dict, Optional, Union


def index_lance(*,
                local: str,
                remote: Optional[str] = None,
                split: Optional[str] = None,
                version: Optional[int] = None,
                download_timeout: Union[float, str] = '1m',
                max_file_bytes: Optional[Union[int, str]] = '200mb',
                columns: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Initialize from a local and/or remote Lance dataset directory.

    Args:
        local (str): Where the dataset is cached on the local filesystem.
        remote (str, optional): Where the dataset is downloaded from. Defaults to ``None``.
        split (str, optional): Which dataset split to use. Defaults to ``None``.
        version (int, optional): Which snapshot version of the dataset to use, or else take
            the latest if ``None``. Defaults to ``None``.
        download_timeout (Union[float, str]): For each Lance metadata file. Defaults to ``1m``.
        max_file_bytes (Union[int, str], optional): File size limit, above which we raise an
            error. This is a performance guard rail, as choppiness increases linearly with
            shard size. The sweet spot is typically around 32mb. Defaults to ``200mb``.
        columns (Dict[str, str], optional): For field names and types specified here, override
            the inferred schema to configure it manually. Defaults to ``None``.

    Returns:
        Dict[str, Any]: StreamingDataset index configuration to stream this Lance dataset.
    """
    raise NotImplementedError  # TODO
