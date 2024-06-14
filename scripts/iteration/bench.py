# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark dataset iteration time."""

import json
import os
from argparse import ArgumentParser, Namespace
from time import time
from typing import Iterator

import lance
import numpy as np
from lance import LanceDataset
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
    args.add_argument('--streaming_dataset', type=str, required=True)
    args.add_argument('--lance_dataset', type=str, required=True)
    args.add_argument('--lance_pow', type=int, default=4)
    args.add_argument('--pq_suffix', type=str, default='.parquet')
    args.add_argument('--tqdm', type=int, default=1)
    args.add_argument('--time_limit', type=float, default=180)
    args.add_argument('--stats', type=str, required=True)
    return args.parse_args()


def bench_lance_seq(dataset: LanceDataset, take_count: int, use_tqdm: int,
                    time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a Lance dataset in sequential order.

    Args:
        dataset (LanceDataset): The Lance dataset to iterate.
        take_count (int): How many samples to take per sequential access.
        use_tqdm (int): Whether to use tqdm.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    num_samples = dataset.count_rows()
    if num_samples % take_count:
        raise ValueError(f'`num_samples` ({num_samples}) must be divisible by `take_count` ' +
                         f'({take_count}).')
    num_batches = num_samples // take_count
    shape = num_batches, take_count
    times = np.zeros(shape, np.float64)
    sample, = dataset.head(1).to_pylist()
    columns = sorted(sample)
    each_batch = enumerate(dataset.to_batches(columns=columns, batch_size=take_count))
    if use_tqdm:
        each_batch = tqdm(each_batch, total=num_batches, leave=False)
    t0 = time()
    for i, samples in each_batch:
        samples.to_pylist()
        assert len(samples) == take_count
        if num_batches < i:  # ???
            break
        times[i] = t = time() - t0
        if time_limit <= t:
            times = times[:i]
            break
    return times.flatten()


def bench_lance_rand(dataset: LanceDataset, take_count: int, use_tqdm: int,
                     time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a Lance dataset in random order.

    Args:
        dataset (LanceDataset): The Lance dataset to iterate.
        take_count (int): How many samples to take per random access.
        use_tqdm (int): Whether to use tqdm.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    num_samples = dataset.count_rows()
    if num_samples % take_count:
        raise ValueError(f'`num_samples` ({num_samples}) must be divisible by `take_count` ' +
                         f'({take_count}).')
    shape = num_samples // take_count, take_count
    times = np.zeros(shape, np.float64)
    batches = np.random.permutation(num_samples).reshape(shape)
    if use_tqdm:
        batches = tqdm(batches, leave=False)
    t0 = time()
    for i, sample_ids in enumerate(batches):
        dataset.take(sample_ids).to_pylist()
        times[i] = t = time() - t0
        if time_limit <= t:
            times = times[:i]
            break
    return times.flatten()


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


def bench_pq_seq(dataset: StreamingDataset, pq_suffix: str, use_tqdm: int,
                 time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in sequential order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        pq_suffix (str): Parquet shard file suffix.
        use_tqdm (int): Whether to use tqdm.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    times = np.zeros(dataset.num_samples, np.float64)
    pbar = tqdm(total=dataset.num_samples, leave=False) if use_tqdm else None
    i = 0
    dataset_root = dataset.streams[0].local
    t0 = time()
    for file in each_pq(dataset_root, pq_suffix):
        table = pq.read_table(file)
        for _ in table.to_pylist():
            times[i] = t = time() - t0
            if time_limit <= t:
                return times[:i]
            i += 1
            if pbar:
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
    pbar = tqdm(total=dataset.num_samples, leave=False) if use_tqdm else None
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
        if pbar:
            pbar.update(1)
    return times


def bench_pq_rand_uncached(dataset: StreamingDataset, pq_suffix: str, use_tqdm: int,
                           time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in random order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        pq_suffix (str): Parquet shard file suffix.
        use_tqdm (int): Whether to use tqdm.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    dataset_root = dataset.streams[0].local
    shard_files = list(each_pq(dataset_root, pq_suffix))
    indices = np.random.permutation(dataset.num_samples)
    times = np.zeros(dataset.num_samples, np.float64)
    pbar = tqdm(total=dataset.num_samples, leave=False) if use_tqdm else None
    t0 = time()
    for i, sample_id in enumerate(indices):
        shard_id, shard_sample_id = dataset.spanner[sample_id]
        shard_file = shard_files[shard_id]
        table = pq.read_table(shard_file)
        shard_samples = table.to_pylist()
        shard_samples[shard_sample_id]
        times[i] = t = time() - t0
        if pbar:
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


def bench_seq(dataset: StreamingDataset, use_tqdm: int, time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in sequential order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        use_tqdm (int): Whether to use tqdm.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    times = np.zeros(dataset.num_samples, np.float64)
    xrange = trange(dataset.num_samples, leave=False) if use_tqdm else range(dataset.num_samples)
    t0 = time()
    for i in xrange:
        dataset[i]
        times[i] = t = time() - t0
        if time_limit <= t:
            times = times[:i]
            break
    return times


def bench_rand(dataset: StreamingDataset, use_tqdm: int, time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in random order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        use_tqdm (int): Whether to use tqdm.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    indices = np.random.permutation(dataset.num_samples)
    times = np.zeros(dataset.num_samples)
    if use_tqdm:
        indices = tqdm(indices, leave=False)
    t0 = time()
    for i, sample_id in enumerate(indices):
        dataset[sample_id]
        times[i] = t = time() - t0
        if time_limit <= t:
            times = times[:i]
            break
    return times


def main(args: Namespace) -> None:
    """Randomly iterate over a Parquet dataset with Streaming.

    Args:
        args (Namespace): Command-line arguments.
    """
    streaming_dataset = StreamingDataset(local=args.streaming_dataset)
    lance_dataset = lance.dataset(args.lance_dataset)

    if args.lance_pow == 4:
        lance_take_counts = 1, 4, 16, 64, 256, 1024
    elif args.lance_pow == 2:
        lance_take_counts = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    else:
        raise ValueError(f'Unsupported --lance_pow: {args.lance_pow}.')

    obj = {}

    to_dict = lambda label, rate, times: ({
        'label': label,
        'rate': rate,
        'times': (times * 1e9).astype(np.int64).tolist()
    })

    times = bench_pq_seq(streaming_dataset, args.pq_suffix, args.tqdm, args.time_limit)
    rate = int(len(times) / times[-1])
    label = f'PQ seq (in mem): {rate:,}/s'
    obj['pq_seq'] = to_dict(label, rate, times)
    print(label)

    times = bench_pq_rand_uncached(streaming_dataset, args.pq_suffix, args.tqdm, args.time_limit)
    rate = int(len(times) / times[-1])
    label = f'PQ rand (in mem): {rate:,}/s'
    obj['pq_rand'] = to_dict(label, rate, times)
    print(label)

    clear_mds(args.streaming_dataset)

    times = bench_seq(streaming_dataset, args.tqdm, args.time_limit)
    rate = int(len(times) / times[-1])
    label = f'Cold PQ>MDS seq: {rate:,}/s'
    obj['pq_mds_seq'] = to_dict(label, rate, times)
    print(label)

    clear_mds(args.streaming_dataset)

    times = bench_rand(streaming_dataset, args.tqdm, args.time_limit)
    rate = int(len(times) / times[-1])
    label = f'Cold PQ>MDS rand: {rate:,}/s'
    obj['pq_mds_rand'] = to_dict(label, rate, times)
    print(label)

    times = bench_seq(streaming_dataset, args.tqdm, args.time_limit)
    rate = int(len(times) / times[-1])
    label = f'Warm MDS seq: {rate:,}/s'
    obj['mds_seq'] = to_dict(label, rate, times)
    print(label)

    times = bench_rand(streaming_dataset, args.tqdm, args.time_limit)
    rate = int(len(times) / times[-1])
    label = f'Warm MDS rand: {rate:,}/s'
    obj['mds_rand'] = to_dict(label, rate, times)
    print(label)

    for take_count in lance_take_counts:
        times = bench_lance_seq(lance_dataset, take_count, args.tqdm, args.time_limit)
        rate = int(len(times) / times[-1])
        label = f'Lance seq n={take_count:04}: {rate:,}/s'
        obj[f'lance_seq_{take_count:04}'] = to_dict(label, rate, times)
        print(label)

    for take_count in lance_take_counts:
        times = bench_lance_rand(lance_dataset, take_count, args.tqdm, args.time_limit)
        rate = int(len(times) / times[-1])
        label = f'Lance rand n={take_count:04}: {rate:,}/s'
        obj[f'lance_rand_{take_count:04}'] = to_dict(label, rate, times)
        print(label)

    with open(args.stats, 'w') as out:
        json.dump(obj, out)


if __name__ == '__main__':
    main(parse_args())
