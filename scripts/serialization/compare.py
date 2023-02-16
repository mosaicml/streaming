# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Compare dataset serialization methods.

Serialization methods compared:
* Arrow https://arrow.apache.org/
* MDS https://github.com/mosaicml/streaming
* Parquet https://parquet.apache.org/

We generate datasets containing identical samples in each format, and compare the time it takes to
iterate over them in three ways:
* sequential: __iter__ on dataset (or sequential __getitem__ if not available)
* shuffled: __iter__ on dataset shuffled idiomatically
* random: iterate in random order via __getitem__

We find that MDS has excellent shuffled iteration performance, identical to iterating sequentially.
This is critical because machine learning training jobs (its intended use case) iterate over the
training split in shuffled order.
"""

import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob
from shutil import rmtree
from time import time
from typing import Any, Callable, Dict, Iterator, List

import numpy as np
import pandas as pd
from datasets import Dataset, disable_caching, load_dataset, load_from_disk  # pyright: ignore
from matplotlib import pyplot as plt
from tqdm import tqdm

from streaming import MDSWriter, StreamingDataset

disable_caching()


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--data',
                      type=str,
                      default='/tmp/streaming/perf_test/',
                      help='Benchmarking datasets root directory')
    args.add_argument('--num_samples', type=int, default=200_000, help='Dataset size')
    args.add_argument('--out', type=str, default='plot.png', help='Path to output plot')
    return args.parse_args()


def fetch_dataset(data_root: str, num_samples: int) -> List[Dict[str, Any]]:
    """Fetch dataset to benchmark as JSONL.

    Args:
        data_root (str): Directory to cache samples in.
        num_samples (int): Only use the first this many samples.

    Returns:
        List[Dict[str, Any]]: The samples.
    """
    if not os.path.isdir(data_root):
        os.makedirs(data_root)
    filename = os.path.join(data_root, 'samples.jsonl')

    samples = []
    if os.path.exists(filename):
        for i, line in tqdm(enumerate(open(filename)), total=num_samples, leave=False):
            if i == num_samples:
                break
            sample = json.loads(line)
            samples.append(sample)

        if len(samples) != num_samples:
            samples = []
            os.remove(filename)

    if not os.path.exists(filename):
        dataset = load_dataset('the_pile', 'pubmed_central', split='train', streaming=True)
        with open(filename, 'w') as out:
            for i, sample in tqdm(enumerate(dataset), total=num_samples, leave=False):
                if i == num_samples:
                    break
                samples.append(sample)
                line = json.dumps(sample, sort_keys=True) + '\n'
                out.write(line)

    return samples


def arrow_write(samples: List[Dict[str, Any]], data_dir: str) -> None:
    """Serialize dataset in arrow format.

    Args:
        samples (List[Dict[str, Any]]): List of sample dicts.
        data_dir (str): Output data directory.
    """
    df = pd.DataFrame(samples)
    ds = Dataset.from_pandas(df)
    ds.save_to_disk(data_dir)


def arrow_read_seq(dirname: str) -> Iterator[Dict[str, Any]]:
    """Iterate an arrow dataset sequentially.

    Args:
        dirname (str): Dataset directory.

    Returns:
        Iterator[Dict[str, Any]]: Iterator over samples.
    """
    dataset = load_from_disk(dirname)
    dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
    yield from dataset.iter(batch_size=1)  # pyright: ignore


def arrow_read_shuf(dirname: str) -> Iterator[Dict[str, Any]]:
    """Iterate an arrow dataset in shuffled order.

    Args:
        dirname (str): Dataset directory.

    Returns:
        Iterator[Dict[str, Any]]: Iterator over samples.
    """
    dataset = load_from_disk(dirname)
    dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
    yield from dataset.shuffle().iter(batch_size=1)  # pyright: ignore


def arrow_read_rand(dirname: str) -> Iterator[Dict[str, Any]]:
    """Iterate an arrow dataset in random order.

    Args:
        dirname (str): Dataset directory.

    Returns:
        Iterator[Dict[str, Any]]: Iterator over samples.
    """
    dataset = load_from_disk(dirname)
    dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
    indices = np.random.permutation(len(dataset))
    for idx in map(int, indices):
        yield dataset[idx]  # pyright: ignore


def mds_write(samples: List[Dict[str, Any]], data_dir: str) -> None:
    """Serialize dataset in mds format.

    Args:
        samples (List[Dict[str, Any]]): List of sample dicts.
        data_dir (str): Output data directory.
    """
    columns = {
        'text': 'str',
        'meta': 'json',
    }
    with MDSWriter(data_dir, columns, size_limit=1 << 26) as out:
        for sample in samples:
            out.write(sample)


def mds_read_seq(dirname: str) -> Iterator[Dict[str, Any]]:
    """Iterate an mds dataset sequentially.

    Args:
        dirname (str): Dataset directory.

    Returns:
        Iterator[Dict[str, Any]]: Iterator over samples.
    """
    yield from StreamingDataset(dirname)


def mds_read_shuf(dirname: str) -> Iterator[Dict[str, Any]]:
    """Iterate an mds dataset in shuffled order.

    Args:
        dirname (str): Dataset directory.

    Returns:
        Iterator[Dict[str, Any]]: Iterator over samples.
    """
    yield from StreamingDataset(dirname, shuffle=True)


def mds_read_rand(dirname: str) -> Iterator[Dict[str, Any]]:
    """Iterate an mds dataset in random order.

    Args:
        dirname (str): Dataset directory.

    Returns:
        Iterator[Dict[str, Any]]: Iterator over samples.
    """
    dataset = StreamingDataset(dirname)
    indices = np.random.permutation(dataset.index.total_samples)
    for idx in indices:
        yield dataset[idx]


def parquet_write(samples: List[Dict[str, Any]], data_dir: str) -> None:
    """Serialize dataset in parquet format.

    Args:
        samples (List[Dict[str, Any]]): List of sample dicts.
        data_dir (str): Output data directory.
    """
    os.makedirs(data_dir)
    df = pd.DataFrame(samples)
    for i in range(0, len(samples), 5_000):
        x = df.iloc[i:i + 5_000]
        x.to_parquet(f'{data_dir}/chunk_{i:06}.parquet')


def parquet_read_seq(dirname: str) -> Iterator[Dict[str, Any]]:
    """Iterate a parquet dataset sequentially.

    Args:
        dirname (str): Dataset directory.

    Returns:
        Iterator[Dict[str, Any]]: Iterator over samples.
    """
    ff = sorted(glob(f'{dirname}/*.parquet'))
    for f in ff:
        df = pd.read_parquet(f)
        for i in range(len(df)):
            yield dict(df.iloc[i])


parquet_read_shuf = None


def parquet_read_rand(dirname: str) -> Iterator[Dict[str, Any]]:
    """Iterate a parquet dataset in random order.

    Args:
        dirname (str): Dataset directory.

    Returns:
        Iterator[Dict[str, Any]]: Iterator over samples.
    """
    ff = sorted(glob(f'{dirname}/*.parquet'))
    dfs = [None] * len(ff)
    pairs = []
    for i, f in enumerate(ff):
        df = pd.read_parquet(f)
        for j in range(len(df)):
            pairs.append((i, j))
    np.random.shuffle(pairs)
    for i, j in pairs:
        df = dfs[i]
        if df is None:
            f = ff[i]
            df = dfs[i] = pd.read_parquet(f)
        yield dict(df.iloc[j])


def bench(read: Callable, dirname: str) -> List[float]:
    """Benchmark iterating over a dataset.

    Args:
        read (Callable): Method that iterates a dataset.
        dirname (str): Dataset directory.

    Returns:
        List[float]: Cumulative time per sample.
    """
    tt = []
    t0 = time()
    for _ in read(dirname):
        t = time() - t0
        tt.append(t)
    return tt


def main(args: Namespace) -> None:
    """Main method, which benchmarks various dataset formats on the same data.

    Args:
        args (Namespace): Command-line arguments.
    """
    print('Fetching dataset...')
    samples = fetch_dataset(args.data, args.num_samples)

    tuples = [
        ('arrow', arrow_write, arrow_read_seq, arrow_read_shuf, arrow_read_rand, 'green'),
        ('mds', mds_write, mds_read_seq, mds_read_shuf, mds_read_rand, 'red'),
        ('parquet', parquet_write, parquet_read_seq, parquet_read_shuf, parquet_read_rand, 'blue'),
    ]

    for name, write, read_seq, read_shuf, read_rand, color in tuples:
        print(f'Benchmarking {name}...')
        dirname = os.path.join(args.data, name)
        if os.path.exists(dirname):
            rmtree(dirname)
        write(samples, dirname)

        if read_seq:
            seq_times = bench(read_seq, dirname)
            plt.plot(seq_times, ls='-', c=color, label=f'{name} sequential')

        if read_shuf:
            shuf_times = bench(read_shuf, dirname)
            plt.plot(shuf_times, ls='--', c=color, label=f'{name} shuffled')

        if read_rand:
            rand_times = bench(read_rand, dirname)
            plt.plot(rand_times, ls=':', c=color, label=f'{name} random')

    print('Plotting...')
    plt.title('Throughput')
    plt.xlabel('Samples')
    plt.ylabel('Seconds')
    plt.legend()
    plt.savefig(args.out, dpi=400)

    print('Done.')


if __name__ == '__main__':
    main(parse_args())
