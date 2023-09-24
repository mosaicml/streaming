# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Randomly iterate over a Parquet dataset with Streaming."""

import os
from argparse import ArgumentParser, Namespace
from time import time
from typing import Iterator

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pyarrow import parquet as pq
from tqdm import tqdm, trange

from streaming import StreamingDataset


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, required=True)
    args.add_argument('--pq_suffix', type=str, default='.parquet')
    args.add_argument('--tqdm', type=int, default=1)
    args.add_argument('--time_limit', type=float, default=10)
    args.add_argument('--plot', type=str, required=True)
    return args.parse_args()


def each_pq(dataset_root: str, pq_suffix: str) -> Iterator[str]:
    """Iteracte over each Parquet shard file of the dataset in order.

    Args:
        dataset_root (str): Dataset root directory.
        pq_suffix (str): Parquet shard file suffix.

    Returns:
        Iterator[str]: Each Parquet shard file.
    """
    for cwd, _, files in os.walk(dataset_root):
        files = filter(lambda file: file.endswith(pq_suffix), files)
        files = (os.path.join(cwd, file) for file in files)
        yield from sorted(files)


def bench_pq_seq(dataset: StreamingDataset, pq_suffix: str, use_tqdm: int) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in sequential order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        pq_suffix (str): Parquet shard file suffix.
        use_tqdm (int): Whether to use tqdm.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    times = np.zeros(dataset.num_samples, np.float64)
    pbar = tqdm(total=dataset.num_samples) if use_tqdm else None
    i = 0
    dataset_root = dataset.streams[0].local
    t0 = time()
    for file in each_pq(dataset_root, pq_suffix):
        table = pq.read_table(file)
        for _ in table.to_pylist():
            times[i] = time() - t0
            i += 1
            if use_tqdm:
                pbar.update(1)
    return times


def bench_pq_rand_cached(dataset: StreamingDataset, pq_suffix: str,
                         use_tqdm: int) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in random order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        pq_suffix (str): Parquet shard file suffix.
        use_tqdm (int): Whether to use tqdm.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    dataset_root = dataset.streams[0].local
    shard_files = list(each_pq(dataset_root, pq_suffix))
    shard_sample_lists = [None] * len(shard_files)
    indices = np.random.permutation(dataset.num_samples)
    times = np.zeros(dataset.num_samples, np.float64)
    pbar = tqdm(total=dataset.num_samples) if use_tqdm else None
    t0 = time()
    for i, sample_id in enumerate(indices):
        shard_id, shard_sample_id = dataset.spanner[sample_id]
        shard_samples = shard_sample_lists[shard_id]
        if shard_samples is None:
            shard_file = shard_files[shard_id]
            table = pq.read_table(shard_file)
            shard_sample_lists[shard_id] = shard_samples = table.to_pylist()
        shard_samples[shard_sample_id]
        times[i] = time() - t0
        if use_tqdm:
            pbar.update(1)
    return times


def bench_pq_rand_uncached(dataset: StreamingDataset, pq_suffix: str, use_tqdm: int,
                           time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in random order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        pq_suffix (str): Parquet shard file suffix.
        use_tqdm (int): Whether to use tqdm.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    dataset_root = dataset.streams[0].local
    shard_files = list(each_pq(dataset_root, pq_suffix))
    indices = np.random.permutation(dataset.num_samples)
    times = np.zeros(dataset.num_samples, np.float64)
    pbar = tqdm(total=dataset.num_samples) if use_tqdm else None
    t0 = time()
    for i, sample_id in enumerate(indices):
        shard_id, shard_sample_id = dataset.spanner[sample_id]
        shard_file = shard_files[shard_id]
        table = pq.read_table(shard_file)
        shard_samples = table.to_pylist()
        shard_samples[shard_sample_id]
        times[i] = t = time() - t0
        if use_tqdm:
            pbar.update(1)
        if time_limit <= t:
            times = times[:i]
            break
    return times


def clear_mds(dataset_root: str) -> None:
    """Clear the intermediate MDS shard files.

    Args:
        dataset_root (str): Dataset root directoyr.
    """
    for cwd, _, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.mds'):
                file = os.path.join(cwd, file)
                os.remove(file)


def bench_seq(dataset: StreamingDataset, use_tqdm: int) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in sequential order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        use_tqdm (int): Whether to use tqdm.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    times = np.zeros(dataset.num_samples, np.float64)
    t0 = time()
    xrange = trange if use_tqdm else range
    for i in xrange(dataset.num_samples):
        dataset[i]
        times[i] = time() - t0
    return times


def bench_rand(dataset: StreamingDataset, use_tqdm: int) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in random order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        use_tqdm (int): Whether to use tqdm.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    indices = np.random.permutation(dataset.num_samples)
    times = np.zeros(dataset.num_samples)
    t0 = time()
    if use_tqdm:
        indices = tqdm(indices)
    for i, sample_id in enumerate(indices):
        dataset[sample_id]
        times[i] = time() - t0
    return times


def main(args: Namespace) -> None:
    """Randomly iterate over a Parquet dataset with Streaming.

    Args:
        args (Namespace): Command-line arguments.
    """
    dataset = StreamingDataset(local=args.dataset)

    plt.title('Time to iterate')
    plt.xlabel('Seconds')
    plt.ylabel('Samples')
    samples = np.arange(dataset.num_samples)

    times = bench_pq_seq(dataset, args.pq_suffix, args.tqdm)
    rate = int(len(times) / times[-1])
    plt.plot(times, samples, c='green', ls='--', label=f'PQ seq (in mem): {rate:,}/s')

    times = bench_pq_rand_uncached(dataset, args.pq_suffix, args.tqdm, args.time_limit)
    rate = int(len(times) / times[-1])
    plt.plot(times, samples[:len(times)], c='green', ls=':',
             label=f'PQ rand (in mem): {rate:,}/s')

    clear_mds(args.dataset)
    times = bench_seq(dataset, args.tqdm)
    rate = int(len(times) / times[-1])
    plt.plot(times, samples, c='blue', ls='--', label=f'Cold PQ>MDS seq: {rate:,}/s')

    clear_mds(args.dataset)
    times = bench_rand(dataset, args.tqdm)
    rate = int(len(times) / times[-1])
    plt.plot(times, samples, c='blue', ls=':', label=f'Cold PQ>MDS rand: {rate:,}/s')

    times = bench_seq(dataset, args.tqdm)
    rate = int(len(times) / times[-1])
    plt.plot(times, samples, c='red', ls='--', label=f'Warm MDS seq: {rate:,}/s')

    times = bench_rand(dataset, args.tqdm)
    rate = int(len(times) / times[-1])
    plt.plot(times, samples, c='red', ls=':', label=f'Warm MDS rand: {rate:,}/s')

    plt.legend()
    plt.grid(which='major', ls='--', c='#ddd')
    plt.savefig(args.plot, dpi=500)


if __name__ == '__main__':
    main(parse_args())
