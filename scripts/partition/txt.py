# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line tool to visualize StreamingDataset sample space partitioning."""

import math
from argparse import ArgumentParser, Namespace

from streaming.base.partitioning import get_partitions


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('-n', '--dataset_size', type=int, default=678)
    args.add_argument('-b', '--device_batch_size', type=int, default=7)
    args.add_argument('-o', '--offset_in_epoch', type=int, default=0)
    args.add_argument('-c', '--canonical_nodes', type=int, default=6)
    args.add_argument('-p', '--physical_nodes', type=int, default=3)
    args.add_argument('-d', '--node_devices', type=int, default=4)
    args.add_argument('-w', '--device_workers', type=int, default=5)
    return args.parse_args()


def main(args: Namespace) -> None:
    """Generate and display a partitioning given command-line arguments.

    Args:
        args (Namespace): Command-line arguments.
    """
    ids = get_partitions(args.dataset_size, args.canonical_nodes, args.physical_nodes,
                         args.node_devices, args.device_workers, args.device_batch_size,
                         args.offset_in_epoch)
    ids = ids.reshape(args.physical_nodes, args.node_devices, args.device_workers, -1,
                      args.device_batch_size)
    max_id = ids.max()
    max_digits = math.ceil(math.log10(max_id + 1))
    pre_cols = max(map(len,
                       [f'Node {args.physical_nodes - 1}', f'Device {args.node_devices - 1}']))
    for i, node in enumerate(ids):
        if i:
            print()
            pre = ' ' * pre_cols + ' + '
            _, _, _, batches, samples = ids.shape
            table = []
            for batch in range(batches):
                row = []
                for sample in range(samples):
                    cell = '-' * max_digits
                    row.append(cell)
                row = '-'.join(row)
                table.append(row)
            line = pre + '--'.join(table)
            print(line)
            print()
        for j, device in enumerate(node):
            if j:
                print()
            for k, worker in enumerate(device):
                table = []
                if k == 0:
                    s = f'Node {i}'.ljust(pre_cols)
                elif k == 1:
                    s = f'Device {j}'.ljust(pre_cols)
                else:
                    s = ' ' * pre_cols
                pre = s + ' | '
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
                print(pre + line)


if __name__ == '__main__':
    main(parse_args())
