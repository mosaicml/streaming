# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Randomly iterate over a Parquet dataset with Streaming."""

import os
from argparse import ArgumentParser, Namespace
from time import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from streaming import StreamingDataset


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, required=True)
    args.add_argument('--cache_mds', type=int, default=1)
    args.add_argument('--plot', type=str, required=True)
    return args.parse_args()


def clear(local: str) -> None:
    """Clear the intermediate MDS shard files."""
    for root, _, files in os.walk(local):
        for file in files:
            file = os.path.join(root, file)
            if file.endswith('.mds'):
                os.remove(file)


def main(args: Namespace) -> None:
    """Randomly iterate over a Parquet dataset with Streaming.

    Args:
        args (Namespace): Command-line arguments.
    """
    dataset = StreamingDataset(local=args.dataset)

    if not args.cache_mds:
        clear(args.dataset)

    seq_times = np.zeros(dataset.num_samples)
    t0 = time()
    for i in trange(dataset.num_samples):
        dataset[i]
        seq_times[i] = time() - t0

    if not args.cache_mds:
        clear(args.dataset)

    indices = np.random.permutation(dataset.num_samples)
    rand_times = np.zeros(dataset.num_samples)
    t0 = time()
    for i, index in enumerate(tqdm(indices)):
        dataset[index]
        rand_times[i] = time() - t0

    plt.title('Parquet sample access times')
    plt.xlabel('Samples seen')
    plt.ylabel('Time (seconds)')
    samples = np.arange(dataset.num_samples)
    plt.plot(samples, seq_times, c='blue', label='Sequential')
    plt.plot(samples, rand_times, c='red', label='Random')
    plt.legend()
    plt.savefig(args.plot, dpi=500)


if __name__ == '__main__':
    main(parse_args())
