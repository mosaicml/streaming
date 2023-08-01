# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Utility and helper functions for regression testing."""

import os
import tempfile


def get_local_remote_dir() -> str:
    """Get a local remote directory."""
    tmp_dir = tempfile.gettempdir()
    tmp_remote_dir = os.path.join(tmp_dir, 'regression_remote')
    return tmp_remote_dir
