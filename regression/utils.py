# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for regression testing."""

import os
import tempfile
from typing import Optional

_CLOUD_UPLOAD_LOCATIONS = {'gs': 'gs://mosaicml-composer-tests/streaming/regression/'}


def get_upload_dir(storage: Optional[str]) -> str:
    """Get an upload directory.

    Args:
        storage (str): Type of storage to use.

    Returns:
        str: Upload directory.
    """
    if storage is None:
        return get_local_upload_dir()
    else:
        return _CLOUD_UPLOAD_LOCATIONS[storage]


def get_local_upload_dir() -> str:
    """Get a local upload directory."""
    tmp_dir = tempfile.gettempdir()
    tmp_upload_dir = os.path.join(tmp_dir, 'regression_upload')
    return tmp_upload_dir
