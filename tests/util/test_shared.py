# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory

import pytest

from streaming.constant import RESUME
from streaming.shared.prefix import _get_path
from streaming.util.shared import clean_stale_shared_memory


def test_clean_stale_shared_memory():
    # Create a leaked shared memory
    name = _get_path(0, RESUME)
    _ = BuiltinSharedMemory(name, True, 64)

    # Clean up the stale shared memory
    clean_stale_shared_memory()

    # If clean up is successful, it should raise FileNotFoundError Exception
    with pytest.raises(FileNotFoundError):
        _ = BuiltinSharedMemory(name, False, 64)
