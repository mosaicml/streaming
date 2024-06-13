# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and plot sample access times across kinds of data and formats."""

import os
import string
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from shutil import rmtree
from time import time, time_ns
from typing import Any, Callable, Dict, Iterator, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from numpy.typing import DTypeLike, NDArray
from tqdm import trange

from streaming import CSVWriter, JSONWriter, MDSWriter, StreamingDataset


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--data',
        type=str,
        default='data',
        help='Data root directory',
    )
    args.add_argument(
        '--mds_color',
        type=str,
        default='red',
        help='Color of MDS curves',
    )
    args.add_argument(
        '--jsonl_color',
        type=str,
        default='green',
        help='Color of JSONL curves',
    )
    args.add_argument(
        '--csv_color',
        type=str,
        default='blue',
        help='Color of CSV curves',
    )
    args.add_argument(
        '--rounds',
        type=int,
        default=5,
        help='Number of rounds of benchmarking each dataset, used to get stable numbers',
    )
    args.add_argument(
        '--dataset_size',
        type=int,
        default=200_000,
        help='Dataset size in samples',
    )
    args.add_argument(
        '--text_len',
        type=int,
        default=1024,
        help='Length of each sample text field in ASCII characters',
    )
    args.add_argument(
        '--tokens_per_sample',
        type=int,
        default=2048,
        help='Length of each sample tokens field in tokens',
    )
    args.add_argument(
        '--token_dtype',
        type=str,
        default='uint16',
        help='Data type of each token',
    )
    args.add_argument(
        '--shard_size_limit',
        type=int,
        default=1 << 25,
        help='Shard size limit in bytes',
    )
    args.add_argument(
        '--truncate_highest_frac',
        type=float,
        default=0.0001,
        help='What fraction of sample access times to truncate on the high end when plotting',
    )
    args.add_argument(
        '--pad_lowest_frac',
        type=float,
        default=0.01,
        help='What fraction of the logarithmic range of truncated sample access latencies to ' +
        'pad on the low end when plotting',
    )
    args.add_argument(
        '--plot_bins',
        type=int,
        default=500,
        help='Number of logarithmic buckets in which to gather sample access latencies',
    )
    args.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='Dots per inch of plots',
    )
    return args.parse_args()


@contextmanager
def timed(text: str) -> Iterator:
    """Time some activity within a contextmanager, logging to stdout.

    Args:
        text (str): The name of the activity to log.

    Returns:
        Iterator: The yield.
    """
    t0 = time()
    yield
    t = time() - t0
    print(f'{text}: {t:.3f} s')


def generate_text(dataset_size: int, text_len: int) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Generate dataset of text samples.

    Args:
        dataset_size (int): Number of samples.
        text_len (int): Length of each text field.

    Returns:
        tuple[Dict[str, str], List[Dict[str, Any]]]: Column names/types and each sample.
    """
    columns = {'text': 'str'}
    vocab = np.array(list(map(ord, string.ascii_letters)))
    assert (vocab < 128).all()
    vocab = vocab.astype(np.uint8)
    arr = np.random.choice(vocab, (dataset_size, text_len))
    samples = []
    for sample_id in trange(dataset_size, leave=False):
        text = arr[sample_id].tobytes().decode('utf-8')
        sample = {'text': text}
        samples.append(sample)
    return columns, samples


def generate_tokens(dataset_size: int, tokens_per_sample: int, dtype: DTypeLike) -> \
        tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Generate dataset of token samples.

    Args:
        dataset_size (int): Number of samples.
        tokens_per_sample (int): Length of each text field.
        dtype (DTypeLike): Which dtype to use for the tokens ndarray.

    Returns:
        tuple[Dict[str, str], List[Dict[str, Any]]]: Column names/types and each sample.
    """
    dtype = np.dtype(dtype)
    columns = {'tokens': f'ndarray:{dtype.name}:{tokens_per_sample}'}
    arr = np.random.randint(0, np.iinfo(dtype).max, (dataset_size, tokens_per_sample), dtype=dtype)
    samples = []
    for sample_id in trange(dataset_size, leave=False):
        tokens = arr[sample_id]
        sample = {'tokens': tokens}
        samples.append(sample)
    return columns, samples


def write(writer_class: type, dirname: str, columns: Dict[str, str], shard_size_limit: int,
          samples: List[Dict[str, Any]]) -> None:
    """Given a writer class and information to write, serialize a dataset in that format.

    Args:
        writer_class (type): Which writer to use (MDS, JSONL, etc.).
        dirname (str): Dataset directory.
        columns (Dict[str, str]): Dataset columns.
        shard_size_limit (int): Maximum shard size.
        samples (List[Dict[str, str]]): Samples that will comprise the dataset.
    """
    if os.path.exists(dirname):
        rmtree(dirname, ignore_errors=True)
    with writer_class(out=dirname, columns=columns, size_limit=shard_size_limit) as out:
        for sample in samples:
            out.write(sample)


def iter_seq(dataset: StreamingDataset) -> NDArray[np.int64]:
    """Iterate a dataset in sequential order, collecting latency per sample.

    Args:
        dataset (StreamingDataset): The dataset.

    Returns:
        NDArray[np.int64]: Timings in integer nanoseconds.
    """
    times = np.zeros(dataset.num_samples, np.int64)
    for sample_id in range(dataset.num_samples):
        t0 = time_ns()
        dataset[sample_id]
        times[sample_id] = time_ns() - t0
    return times


def iter_rand(dataset: StreamingDataset) -> NDArray[np.int64]:
    """Iterate a dataset in random order, collecting latency per sample.

    Args:
        dataset (StreamingDataset): The dataset.

    Returns:
        NDArray[np.int64]: Timings in integer nanoseconds.
    """
    times = np.zeros(dataset.num_samples, np.int64)
    for i, sample_id in enumerate(np.random.permutation(dataset.num_samples)):
        t0 = time_ns()
        dataset[sample_id]
        times[i] = time_ns() - t0
    return times


def bench(args: Namespace, bench_name: str, desc: str, generate: Callable,
          formats: List[str]) -> None:
    """Benchmark a type of data across potentially multiple formats.

    Args:
        args (Namespace): Command-line arguments.
        bench_name (str): What to call this benchmark.
        desc (str): Brief description of the data.
        generate (Callable): Method to genereate the dataset.
        formats (List[str]): List of shard formats to benchmark this data in.
    """
    print(f'Bench: {bench_name}')

    format_infos = [
        ('mds', MDSWriter, args.mds_color),
        ('jsonl', JSONWriter, args.jsonl_color),
        ('csv', CSVWriter, args.csv_color),
    ]
    format_infos = list(filter(lambda info: info[0] in formats, format_infos))

    with timed('  Generate'):
        columns, samples = generate(args.dataset_size)

    print('  Write')
    for format_name, writer_class, _ in format_infos:
        dirname = os.path.join(args.data, bench_name, format_name)
        with timed(f'    {format_name.upper()}'):
            write(writer_class, dirname, columns, args.shard_size_limit, samples)

    datasets = []
    for format_name, _, _ in format_infos:
        local = os.path.join(args.data, bench_name, format_name)
        dataset = StreamingDataset(local=local)
        datasets.append(dataset)

    print('  Read')
    seq_lists = [[] for _ in format_infos]
    rand_lists = [[] for _ in format_infos]
    for i in range(args.rounds):
        print(f'    Round {i}/{args.rounds}')
        for j, (format_name, _, _) in enumerate(format_infos):
            dataset = datasets[j]
            print(f'      {format_name.upper()}')
            with timed(f'        Sequential'):
                times = iter_seq(dataset)
            seq_lists[j].append(times)
            with timed(f'        Random'):
                times = iter_rand(dataset)
            rand_lists[j].append(times)

    # Combine results of rounds.
    seqs = []
    rands = []
    for arrs in seq_lists:
        arr = np.concatenate(arrs)
        seqs.append(arr)
    for arrs in rand_lists:
        arr = np.concatenate(arrs)
        rands.append(arr)

    # Display bounds of data.
    print(f'  Plot')
    print(f'    All sequential')
    print(f'      Min: {min(map(min, seqs)) / 1e3:.3f} μs')
    print(f'      Max: {max(map(max, seqs)) / 1e3:.3f} μs')
    print(f'    All random')
    print(f'      Min: {min(map(min, rands)) / 1e3:.3f} μs')
    print(f'      Max: {max(map(max, rands)) / 1e3:.3f} μs')

    # Determine upper bound of plot.
    #
    # We truncate the right side by ``truncate_highest_frac``, which is in terms of probability
    # mass. The data is typically a big spike followed by a long flat tail.
    times = np.concatenate(seqs + rands)
    times.sort()
    min_time = times[0]
    index = int(len(times) * (1 - args.truncate_highest_frac))
    max_time = times[index]

    # Determine lower bound of plot.
    #
    # We pad the left side by ``pad_lowest_frac``, which is in terms of the truncated range. We are
    # adding just a little bit of extra padding on the left side so that the curves tart from the
    # very bottom. The plot is displayed log-scaled, so the padding is calculated in terms of the
    # log-scaled timings.
    log10_min_time = np.log10(min_time)
    log10_max_time = np.log10(max_time)
    log10_span = log10_max_time - log10_min_time
    log10_min_time -= log10_span * args.pad_lowest_frac
    min_time = 10**log10_min_time
    max_time = 10**log10_max_time
    print(f'    Bounds')
    print(f'      Min: {min_time / 1e3:.3f} μs')
    print(f'      Max: {max_time / 1e3:.3f} μs')
    print(f'      Min log10: {log10_min_time:.3f}')
    print(f'      Max log10: {log10_max_time:.3f}')

    order_infos = [
        ('sequential', '-'),
        ('random', ':'),
    ]

    plt.rc('font', size=6)
    plt.title(f'{bench_name.title()} ({desc}) sample access latency')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Latency (μs)')
    plt.ylabel('Probability')
    plt.grid(which='major', ls='-', c='#ddd', lw=0.5)
    plt.grid(which='minor', ls=':', c='#ddd', lw=0.5)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    ax.xaxis.set_tick_params(which='minor', pad=5)
    print('  Stats')
    for (format_name, writer_class, color), seq, rand in zip(format_infos, seqs, rands):
        print(f'    {format_name.upper()}')
        for times, (order_name, line_style) in zip((seq, rand), order_infos):
            # Locate the logarithmic bins.
            x = np.linspace(log10_min_time, log10_max_time, args.plot_bins + 1)[:-1]

            # Adjust half a bin upward so points are in the center of the space covered by bins.
            x += 1 / (2 * args.plot_bins)

            # Convert back to linear scale, so that matplotlib can do the logarithmic conversion.
            x = 10**x

            # Convert to microseconds.
            x /= 1e3

            y = times

            # Convert to powers of ten (nanoseconds).
            y = np.log10(y)

            # Bucket timings logarithmically within the span that will fit on the plot.
            y -= log10_min_time
            y /= log10_span
            y *= args.plot_bins
            y = y.astype(np.int64)

            # Truncate the higest ``args.truncate_highest_frac`` timings because they get further
            # and further spaced as you ascend, which would ruin the plot.
            y = y[np.nonzero(y < args.plot_bins)[0]]

            # Compute bin magnitudes.
            y = np.bincount(y, minlength=args.plot_bins)

            # Convert that to probability mass based on total number of timings.
            y = y.astype(np.float64)
            y /= (args.rounds * args.dataset_size)

            # Plot x -> y.
            label = f'{format_name.upper()} {order_name}'
            plt.plot(x, y, c=color, ls=line_style, label=label, lw=0.5, alpha=0.5)

            # Note stats.
            print(f'      {order_name.title()}')
            print(f'        Min: {times.min() / 1e3:.3f} μs')
            print(f'        Mode: {x[y.argmax()]:.3f} μs')
            print(f'        Median: {np.median(times) / 1e3:.3f} μs')
            print(f'        Mean: {times.mean() / 1e3:.3f} μs')
            print(f'        Max: {times.max() / 1e3:.3f} μs')
    plt.legend()
    filename = os.path.join(args.data, f'{bench_name}.png')
    plt.savefig(filename, dpi=args.dpi)
    plt.clf()


def main(args: Namespace) -> None:
    """Run the benchmarks.

    Args:
        args (Namespace): Command-line arguments.
    """
    gen_text = lambda dataset_size: generate_text(dataset_size, args.text_len)
    token_dtype = getattr(np, args.token_dtype)
    gen_tokens = lambda dataset_size: generate_tokens(dataset_size, args.tokens_per_sample,
                                                      token_dtype)
    bench_infos = [
        ('text', '1024 chr', gen_text, ('mds', 'jsonl', 'csv')),
        ('tokens', '2048 u16', gen_tokens, ('mds',)),
    ]

    for name, desc, generate, formats in bench_infos:
        bench(args, name, desc, generate, formats)


if __name__ == '__main__':
    main(parse_args())
