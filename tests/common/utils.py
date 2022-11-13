# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
from typing import Any, List, Optional

import pytest


@pytest.fixture(scope='function')
def remote_local() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""
    try:
        mock_dir = tempfile.TemporaryDirectory()
        mock_remote_dir = os.path.join(mock_dir.name, 'remote')
        mock_local_dir = os.path.join(mock_dir.name, 'local')
        yield mock_remote_dir, mock_local_dir
    finally:
        mock_dir.cleanup()  # pyright: ignore


@pytest.fixture(scope='function')
def compressed_remote_local() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""
    try:
        mock_dir = tempfile.TemporaryDirectory()
        mock_compressed_dir = os.path.join(mock_dir.name, 'compressed')
        mock_remote_dir = os.path.join(mock_dir.name, 'remote')
        mock_local_dir = os.path.join(mock_dir.name, 'local')
        yield mock_compressed_dir, mock_remote_dir, mock_local_dir
    finally:
        mock_dir.cleanup()  # pyright: ignore


def get_config_in_bytes(format: str,
                        size_limit: int,
                        column_names: List[str],
                        column_encodings: List[str],
                        column_sizes: List[str],
                        compression: Optional[str] = None,
                        hashes: Optional[List[str]] = None):
    hashes = hashes or []
    config = {
        'version': 2,
        'format': format,
        'compression': compression,
        'hashes': hashes,
        'size_limit': size_limit,
        'column_names': column_names,
        'column_encodings': column_encodings,
        'column_sizes': column_sizes
    }
    return json.dumps(config, sort_keys=True).encode('utf-8')
