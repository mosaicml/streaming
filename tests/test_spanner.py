import numpy as np

from streaming.base.spanner import Spanner


def test_spanner():
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

    try:
        spanner[index]
        assert False
    except ValueError:
        pass

