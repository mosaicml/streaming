# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line tool to compare StreamingDataset sample space partitioning performance."""

from argparse import ArgumentParser, Namespace
from time import time

from streaming.base.partition import get_partitions


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--num_canonical_nodes',
        type=int,
        default=8,
        help='Number of canonical nodes',
    )
    args.add_argument(
        '--num_nodes',
        type=int,
        default=8,
        help='Number of physical nodes',
    )
    args.add_argument(
        '--ranks_per_node',
        type=int,
        default=8,
        help='Ranks per node',
    )
    args.add_argument(
        '--workers_per_rank',
        type=int,
        default=8,
        help='Workers per rank',
    )
    args.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size per rank',
    )
    args.add_argument(
        '--sample_in_epoch',
        type=int,
        default=0,
        help='Sample offset in epoch',
    )
    args.add_argument(
        '--min_power',
        type=int,
        default=10,
        help='Minimum dataset size as a power of 2',
    )
    args.add_argument(
        '--max_power',
        type=int,
        default=34,
        help='Maximum dataset size as a power of 2',
    )
    args.add_argument(
        '--power_interval',
        type=int,
        default=4,
        help='Dataset size step as a power of 2',
    )
    args.add_argument(
        '--timeout',
        type=float,
        default=120,
        help='Maximum time to wait for partitioning to complete',
    )
    return args.parse_args()


def main(args: Namespace) -> None:
    """Benchmark slow vs fast partitioning methods.

    Args:
        args (Namespace): Command-line arguments.
    """
    # Starts true, becomes false when partitioning starts to take too long.
    do_orig = True

    print(f'{"power".rjust(5)} {"samples".rjust(14)} {"orig".rjust(10)}')
    for mul_power in range(args.min_power * args.power_interval,
                           args.max_power * args.power_interval + 1):
        power = mul_power / args.power_interval
        num_samples = int(2**power)

        if do_orig:
            start = time()
            get_partitions('orig', num_samples, args.num_canonical_nodes, args.num_nodes,
                           args.ranks_per_node, args.workers_per_rank, args.batch_size,
                           args.sample_in_epoch)
            elapsed = time() - start
            if args.timeout < elapsed:
                do_orig = False
        else:
            elapsed = 0

        print(f'{power:5.2f} {num_samples:14,} {elapsed:10.6f}')


if __name__ == '__main__':
    main(parse_args())
