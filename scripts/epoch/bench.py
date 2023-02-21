# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark time taken to generate the epoch for a given dataset."""

import json
from argparse import ArgumentParser, Namespace
from time import time

import numpy as np

from streaming.base.partitioning import get_partitions
from streaming.base.shuffle import get_shuffle


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--index', type=str, required=True)
    args.add_argument('--shuffle_algo', type=str, default='py2s')
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--sample_in_epoch', type=int, default=0)
    args.add_argument('--num_canonical_nodes', type=int, default=1)
    args.add_argument('--num_nodes', type=int, default=1)
    args.add_argument('--ranks_per_node', type=int, default=8)
    args.add_argument('--workers_per_rank', type=int, default=8)
    return args.parse_args()


def main(args: Namespace) -> None:
    """Benchmark time taken to generate the epoch for a given dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    obj = json.load(open(args.index))
    shards = obj['shards']
    shard_sizes = [shard['samples'] for shard in shards]
    shard_sizes = np.array(shard_sizes)
    num_samples = sum(shard_sizes)

    t0 = time()
    ids = get_partitions(num_samples, args.num_canonical_nodes, args.num_nodes,
                         args.ranks_per_node, args.workers_per_rank, args.batch_size,
                         args.sample_in_epoch)
    t_part = time() - t0
    print(f'Partition: {t_part:.3f} sec')

    t0 = time()
    mapping = get_shuffle(args.shuffle_algo, shard_sizes, args.num_canonical_nodes, 9176, 0)
    t_shuf = time() - t0
    print(f'Shuffle: {t_shuf:.3f} sec')

    t0 = time()
    ids = np.where(ids == -1, -1, mapping[ids])
    t_remap = time() - t0
    print(f'Remap: {t_remap:.3f} sec')

    t0 = time()
    ids.tofile('tmp.bin')
    t_write = time() - t0
    print(f'Write: {t_write:.3f} sec')

    t0 = time()
    ids = np.fromfile('tmp.bin', np.int64)
    t_read = time() - t0
    print(f'Read: {t_read:.3f} sec')

    t = t_part + t_shuf + t_remap + t_write + t_read
    print(f'All: {t:.3f} sec')


if __name__ == '__main__':
    main(parse_args())
