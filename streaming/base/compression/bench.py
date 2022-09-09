# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from time import time
from typing import Iterator

import numpy as np

from streaming.base.compression.compression import compress, decompress, get_compressions


def parse_args() -> Namespace:
    """Parse commandline arguments.

    Args:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--data', type=str, required=True)
    args.add_argument('--min_power', type=int, default=10)
    args.add_argument('--max_power', type=int, default=30)
    args.add_argument('--max_time', type=float, default=10)
    return args.parse_args()


def each_size(max_size: int, min_power: int, max_power: int) -> Iterator[int]:
    """Get some even exponentially growing sizes of data to benchmark on.

    Args:
        max_size (int): Size of source data.
        min_power (int): Minimum power of two size.
        max_power (int): Maximum power of two size.

    Returns:
        Iterator[int]: Each size.
    """
    for power in range(min_power, max_power + 1):
        for mul in [1, 1.5]:
            size = 1 << power
            size = int(size * mul)
            if max_size < size:
                return
            yield size


def main(args: Namespace) -> None:
    """Benchmark compression algorithms.

    Args:
        args (Namespace): Commandline flags.
    """
    data = open(args.data, 'rb').read()
    for algo in sorted(get_compressions()):
        for size in each_size(len(data), args.min_power, args.max_power):
            i = np.random.choice(len(data) - size + 1)
            s = data[i:i + size]

            t0 = time()
            s_enc = compress(algo, s)
            t_enc = time() - t0

            t0 = time()
            s_dec = decompress(algo, s_enc)
            t_dec = time() - t0

            assert s == s_dec

            print(f'{algo},{size},{len(s_enc)},{t_enc:.9f},{t_dec:.9f}')

            if args.max_time < t_enc + t_dec:
                break


if __name__ == '__main__':
    main(parse_args())
