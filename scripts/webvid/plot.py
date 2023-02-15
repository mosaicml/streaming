# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Plot download performance comparison."""

from argparse import ArgumentParser, Namespace

import numpy as np
from matplotlib import pyplot as plt


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--out', type=str, required=True, help='Where to save the generated plot')
    args.add_argument('--inside',
                      type=str,
                      required=True,
                      help='Path to baseline "inside" benchmark data')
    args.add_argument('--outside_gi',
                      type=str,
                      required=True,
                      help='Path to "outside (__getitem__)" benchmark data')
    args.add_argument('--outside_dt',
                      type=str,
                      required=True,
                      help='Path to "outside (_download_thread) benchmark data')
    return args.parse_args()


def main(args: Namespace) -> None:
    """Plot WebVid download benchmarking results.

    Args:
        args (Namespace): Command-line arguments.
    """
    y = np.fromfile(args.inside, np.float32)
    plt.plot(y, c='black', label='Baseline: videos inside shards')
    x = 1 + np.arange(len(y))
    r = (y / x).mean()
    print(f'Videos inside shards: {r:.6f} seconds/sample (calc from {len(y)} samples)')

    y = np.fromfile(args.outside_gi, np.float32)
    plt.plot(y, c='green', label='Videos separate (__getitem__)')
    x = 1 + np.arange(len(y))
    r = (y / x).mean()
    print(f'Videos separate (__getitem__): {r:.6f} seconds/sample (calc from {len(y)} samples)')

    y = np.fromfile(args.outside_dt, np.float32)
    plt.plot(y, c='blue', label='Videos separate (_download_thread)')
    x = 1 + np.arange(len(y))
    r = (y / x).mean()
    print(
        f'Videos separate (_download_thread): {r:.6f} seconds/sample (calc from {len(y)} samples)')

    plt.title('Sample download rate across the epoch')
    plt.xlabel('Samples')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.savefig(args.out, dpi=400)


if __name__ == '__main__':
    main(parse_args())
