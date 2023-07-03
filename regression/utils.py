# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for regression testing."""

import os
import tempfile
from typing import Optional

_CLOUD_REMOTE_LOCATIONS = {'gs': 'gs://mosaicml-composer-tests/streaming/regression/'}


def get_remote_dir(storage: Optional[str]) -> str:
    """Get an remote directory.

    Args:
        storage (str): Type of storage to use.

    Returns:
        str: Remote directory.
    """
    if storage is None:
        return get_local_remote_dir()
    else:
        return _CLOUD_REMOTE_LOCATIONS[storage]


def get_local_remote_dir() -> str:
    """Get a local remote directory."""
    tmp_dir = tempfile.gettempdir()
    tmp_remote_dir = os.path.join(tmp_dir, 'regression_remote')
    return tmp_remote_dir
