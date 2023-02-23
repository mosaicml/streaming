# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line tool to visualize StreamingDataset sample space partitioning."""

import math
from argparse import ArgumentParser, Namespace

import numpy as np
from numpy.typing import NDArray

from streaming.base.partition import get_partitions


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('-a', '--algo', type=str, default='pynum')
    args.add_argument('-n', '--dataset_size', type=int, default=678)
    args.add_argument('-b', '--device_batch_size', type=int, default=7)
    args.add_argument('-o', '--offset_in_epoch', type=int, default=0)
    args.add_argument('-c', '--canonical_nodes', type=int, default=6)
    args.add_argument('-p', '--physical_nodes', type=int, default=3)
    args.add_argument('-d', '--node_devices', type=int, default=4)
    args.add_argument('-w', '--device_workers', type=int, default=5)
    return args.parse_args()


def show(ids: NDArray[np.int64]) -> None:
    """Display a sample ID tensor on stdout.

    Args:
        ids (NDArray[np.int64]): Sample ID tensor of shape (node, rank, worker, batch, sample).
    """
    max_id = ids.max()
    max_digits = math.ceil(math.log10(max_id + 1))
    for i, node in enumerate(ids):
        print(f'Node {i}')
        for j, device in enumerate(node):
            print(f'    Dev {j}')
            for worker in device:
                table = []
                for batch in worker:
                    row = []
                    for sample in batch:
                        if 0 <= sample:
                            cell = str(sample)
                        else:
                            cell = '-'
                        cell = cell.rjust(max_digits)
                        row.append(cell)
                    row = ' '.join(row)
                    table.append(row)
                line = '  '.join(table)
                print(' ' * 12 + line)


def main(args: Namespace) -> None:
    """Generate and display a partitioning given command-line arguments.

    Args:
        args (Namespace): Command-line arguments.
    """
    ids = get_partitions(args.algo, args.dataset_size, args.canonical_nodes, args.physical_nodes,
                         args.node_devices, args.device_workers, args.device_batch_size,
                         args.offset_in_epoch)
    show(ids)


if __name__ == '__main__':
    main(parse_args())
