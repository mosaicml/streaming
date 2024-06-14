# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from streaming.base.spanner import Spanner


def test_spanner_success():
    shard_sizes = np.arange(5, 100, 5)
    span_size = 7
    spanner = Spanner(shard_sizes, span_size)
    index = 0
    for wanted_shard_id, shard_size in enumerate(shard_sizes):
        for wanted_offset in range(shard_size):
            got_shard_id, got_offset = spanner[index]
            assert got_shard_id == wanted_shard_id
            assert got_offset == wanted_offset
            index += 1


@pytest.mark.parametrize('index', [-10, 2000])
def test_spanner_invalid_index(index: int):
    shard_sizes = np.arange(5, 100, 5)
    span_size = 7
    with pytest.raises(IndexError, match='Invalid sample index.*'):
        spanner = Spanner(shard_sizes, span_size)
        spanner[index]
