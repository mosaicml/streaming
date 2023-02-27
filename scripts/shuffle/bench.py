# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line tool to benchmark StreamingDataset shuffling performance."""

import math
from argparse import ArgumentParser, Namespace
from time import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from streaming.base.shuffle import get_shuffle_py1s, get_shuffle_py2s


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--num_canonical_nodes',
                      type=int,
                      default=1,
                      help='Number of canonical nodes')
    args.add_argument('--seed', type=int, default=9186, help='Shuffle seed')
    args.add_argument('--epoch', type=int, default=0, help='Current epoch')
    args.add_argument('--timeout',
                      type=float,
                      default=120,
                      help='The longest that we are willing to wait for results, in seconds')
    args.add_argument('--samples_per_shard', type=float, default=50_000, help='Samples per shard')
    args.add_argument('--min_power', type=int, default=10, help='Min dataset size as a power of 2')
    args.add_argument('--max_power', type=int, default=34, help='Max dataset size as a power of 2')
    args.add_argument('--power_interval',
                      type=int,
                      default=4,
                      help='Dataset size step as a power of 2')
    return args.parse_args()


class Caller(object):
    """Partial function application with timeouts.

    Args:
        func (Callable): The wrapped function to time.
        num_canonical_nodes (int): Shuffle canonical nodes.
        seed (int): Shuffle seed.
        epoch (int): Shuffle epoch.
        timeout (float): Maximum time elapsed before it skips benchmarking the wrapped method.
    """

    def __init__(self, func: Callable, num_canonical_nodes: int, seed: int, epoch: int,
                 timeout: float) -> None:
        self.func = func
        self.num_canonical_nodes = num_canonical_nodes
        self.seed = seed
        self.epoch = epoch
        self.timeout = timeout
        self.do = True

    def __call__(self, shard_sizes: NDArray[np.int64]) -> float:
        """Call the contained function with the given shard sizes.

        Args:
            shard_sizes (NDArray[np.int64]): Size of each shard.

        Returns:
            float: Time to shuffle.
        """
        if not self.do:
            return 0
        start = time()
        self.func(shard_sizes, self.num_canonical_nodes, self.seed, self.epoch)
        elapsed = time() - start
        if self.timeout < elapsed:
            self.do = False
        return elapsed


def get_shard_sizes(dataset_size: int, samples_per_shard: float) -> NDArray[np.int64]:
    """Calculate realistic shard sizes given samples and samples/shard.

    Args:
        dataset_size (int): Total samples in the dataset.
        samples_per_shard (float): Average shard size in samples.

    Returns:
        NDArray[np.int64]: The sample size of each shard.
    """
    num_shards = math.ceil(dataset_size / samples_per_shard)
    shard_sizes = num_shards * [samples_per_shard]
    shard_sizes[-1] -= num_shards * samples_per_shard - dataset_size
    return np.array(shard_sizes)


def main(args: Namespace) -> None:
    """Benchmark streaming dataset shuffling methods.

    Args:
        args (Namespace): Command-line arguments.
    """
    names = 'py1s', 'py2s'
    get_shuffles = get_shuffle_py1s, get_shuffle_py2s

    def wrap(func: Callable):
        return Caller(func, args.num_canonical_nodes, args.seed, args.epoch, args.timeout)

    callers = list(map(wrap, get_shuffles))

    text = ' '.join(map(lambda s: s.rjust(10), names))

    print(f'{"power".rjust(5)} {"samples".rjust(14)} ' + text)
    for mul_power in range(args.min_power * args.power_interval,
                           args.max_power * args.power_interval + 1):
        power = mul_power / args.power_interval
        dataset_size = int(2**power)
        shard_sizes = get_shard_sizes(dataset_size, args.samples_per_shard)

        texts = []
        for caller in callers:
            elapsed = caller(shard_sizes)
            text = f'{elapsed:10.6f}'
            texts.append(text)
        text = ' '.join(texts)

        print(f'{power:5.2f} {dataset_size:14,} {text}')


if __name__ == '__main__':
    main(parse_args())
