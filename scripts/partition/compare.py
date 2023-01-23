# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line tool to compare StreamingDataset sample space partitioning performance."""

from argparse import ArgumentParser, Namespace
from time import time

from streaming.base.partitioning import get_partitions_fast, get_partitions_slow


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--num_canonical_nodes', type=int, default=1)
    args.add_argument('--num_nodes', type=int, default=1)
    args.add_argument('--ranks_per_node', type=int, default=8)
    args.add_argument('--workers_per_rank', type=int, default=8)
    args.add_argument('--batch_size', type=int, default=256)
    args.add_argument('--sample_in_epoch', type=int, default=0)
    args.add_argument('--min_power', type=int, default=20)
    args.add_argument('--max_power', type=int, default=34)
    args.add_argument('--power_interval', type=int, default=4)
    return args.parse_args()


def main(args: Namespace) -> None:
    """Benchmark slow vs fast partitioning methods.

    Args:
        args (Namespace): Command-line arguments.
    """
    timeout = 60
    do_slow = True
    do_fast = True

    print(f'{"power".rjust(5)} {"samples".rjust(14)} {"slow".rjust(7)} {"fast".rjust(7)} ' +
          f'{"ratio".rjust(7)}')
    for mul_power in range(args.min_power * args.power_interval,
                           args.max_power * args.power_interval + 1):
        power = mul_power / args.power_interval
        num_samples = int(2**power)

        if do_slow:
            start = time()
            get_partitions_slow(num_samples, args.num_canonical_nodes, args.num_nodes,
                                args.ranks_per_node, args.workers_per_rank, args.batch_size,
                                args.sample_in_epoch)
            elapsed = time() - start
            if timeout < elapsed:
                do_slow = False
        else:
            elapsed = 0

        if do_fast:
            start2 = time()
            get_partitions_fast(num_samples, args.num_canonical_nodes, args.num_nodes,
                                args.ranks_per_node, args.workers_per_rank, args.batch_size,
                                args.sample_in_epoch)
            elapsed2 = time() - start2
            if timeout < elapsed2:
                do_fast = False
        else:
            elapsed2 = 0

        if elapsed and elapsed2:
            ratio = elapsed / elapsed2
        else:
            ratio = 0

        print(f'{power:5.2f} {num_samples:14,} {elapsed:7.3f} {elapsed2:7.3f} {ratio:7.3f}')


if __name__ == '__main__':
    main(parse_args())
