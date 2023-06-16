# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import socket
import tempfile
from typing import Any, List, Optional

import pytest


@pytest.fixture(scope='function')
def local_remote_dir() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""
    try:
        mock_dir = tempfile.TemporaryDirectory()
        mock_local_dir = os.path.join(mock_dir.name, 'local')
        mock_remote_dir = os.path.join(mock_dir.name, 'remote')
        yield mock_local_dir, mock_remote_dir
    finally:
        mock_dir.cleanup()  # pyright: ignore


@pytest.fixture(scope='function')
def compressed_local_remote_dir() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""
    try:
        mock_dir = tempfile.TemporaryDirectory()
        mock_compressed_dir = os.path.join(mock_dir.name, 'compressed')
        mock_local_dir = os.path.join(mock_dir.name, 'local')
        mock_remote_dir = os.path.join(mock_dir.name, 'remote')
        yield mock_compressed_dir, mock_local_dir, mock_remote_dir
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


def get_free_tcp_port() -> int:
    """Get a free socket port to listen on."""
    tcp = socket.socket()
    tcp.bind(('', 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def copy_all_files(source: str, destination: str) -> None:
    """Copy all the files from source directory to destination directory.

    Args:
        source (str): Source directory path.
        destination (str): Destination directory path.
    """
    files = os.listdir(source)
    if not os.path.exists(destination):
        os.mkdir(destination)

    # iterating over all the files in the source directory
    for filename in files:
        # copying the files to the destination directory
        source_filename = os.path.join(source, filename)
        destination_filename = os.path.join(destination, filename)
        if os.path.isfile(source_filename):
            shutil.copy2(source_filename, destination_filename)
