# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from streaming.base.shared import SharedArray


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
def test_shared_array_size_is_integer(dtype):
    shared_array = SharedArray(shape=(2, 3), dtype=dtype, name='test_shared_array')
    assert isinstance(shared_array.shm.size, int), 'Size is not an integer'
