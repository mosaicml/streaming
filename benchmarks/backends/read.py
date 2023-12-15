# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark dataset iteration time."""

import json
import os
from argparse import ArgumentParser, Namespace
from time import time
from typing import Any, Dict, Iterator, List, Tuple

import lance
import numpy as np
from lance import LanceDataset
from numpy.typing import NDArray
from pyarrow import parquet as pq
from pyarrow.parquet import ParquetFile
from tqdm import tqdm, trange

from streaming import StreamingDataset


def _parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--data_root', type=str, default='data/backends/gold/')
    args.add_argument('--split', type=str, default='medium')
    args.add_argument('--lance_pow_interval', type=int, default=4)
    args.add_argument('--parquet_suffix', type=str, default='.parquet')
    args.add_argument('--progress_bar', type=int, default=1)
    args.add_argument('--time_limit', type=float, default=180)
    args.add_argument('--out', type=str, default='data/backends/stats.json')
    return args.parse_args()


def _bench_lance_seq(dataset: LanceDataset, take_count: int, show_progress: bool,
                     time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a Lance dataset in sequential order.

    Args:
        dataset (LanceDataset): The Lance dataset to iterate.
        take_count (int): How many samples to take per sequential access.
        show_progress (bool): Whether to show a progress bar.
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
    if show_progress:
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


def _bench_lance_rand(dataset: LanceDataset, take_count: int, show_progress: bool,
                      time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a Lance dataset in random order.

    Args:
        dataset (LanceDataset): The Lance dataset to iterate.
        take_count (int): How many samples to take per random access.
        show_progress (bool): Whether to show a progress bar.
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
    if show_progress:
        batches = tqdm(batches, leave=False)
    t0 = time()
    for i, sample_ids in enumerate(batches):
        dataset.take(sample_ids).to_pylist()
        times[i] = t = time() - t0
        if time_limit <= t:
            times = times[:i]
            break
    return times.flatten()


def _each_parquet(dataset_root: str, parquet_suffix: str) -> Iterator[str]:
    """Iteracte over each Parquet shard file of the dataset in order.

    Args:
        dataset_root (str): Dataset root directory.
        parquet_suffix (str): Parquet shard file suffix.

    Returns:
        Iterator[str]: Each Parquet shard file.
    """
    ret = []
    for parent, _, file_basenames in os.walk(dataset_root):
        file_basenames = filter(lambda basename: basename.endswith(parquet_suffix), file_basenames)
        ret += [os.path.join(parent, basename) for basename in file_basenames]
    yield from sorted(ret)


def _get_parquet_mapping(dataset_dir: str, parquet_suffix: str) -> \
        Tuple[List[str], NDArray[np.int64]]:
    """Get a mapping of sample ID to (shard ID, relative sample ID within that shard).

    Args:
        dataset_dir (str): Parquet dataset directory.
        parquet_suffix (str): Parquet shard file suffix.

    Returns:
        Tuple[List[str], NDArray[np.int64]]: Filenames and mapping.
    """
    filenames = list(_each_parquet(dataset_dir, parquet_suffix))
    mapping = []
    for file_id, filename in enumerate(filenames):
        file = ParquetFile(filename)
        mapping += list(zip([file_id] * file.metadata.num_rows, range(file.metadata.num_rows)))
    mapping = np.array(mapping)
    return filenames, mapping


def _bench_parquet_seq(dataset_dir: str, parquet_suffix: str, show_progress: bool,
                       time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in sequential order.

    Args:
        dataset_dir (str): Parquet dataset directory.
        parquet_suffix (str): Parquet shard file suffix.
        show_progress (bool): Whether to show a progress bar.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    filenames, mapping = _get_parquet_mapping(dataset_dir, parquet_suffix)
    num_samples = len(mapping)
    times = np.zeros(num_samples, np.float64)
    progress_bar = tqdm(total=num_samples, leave=False) if show_progress else None
    i = 0
    t0 = time()
    for filename in filenames:
        table = pq.read_table(filename)
        for _ in table.to_pylist():
            times[i] = t = time() - t0
            if time_limit <= t:
                return times[:i]
            i += 1
            if progress_bar:
                progress_bar.update(1)
    return times


def _bench_parquet_rand(dataset_dir: str, parquet_suffix: str, show_progress: bool,
                        time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in random order.

    Args:
        dataset_dir (str): Parquet dataset directory.
        parquet_suffix (str): Parquet shard file suffix.
        show_progress (bool): Whether to show a progress bar.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    filenames, mapping = _get_parquet_mapping(dataset_dir, parquet_suffix)
    num_samples = len(mapping)
    indices = np.random.permutation(num_samples)
    times = np.zeros(num_samples, np.float64)
    progress_bar = tqdm(total=num_samples, leave=False) if show_progress else None
    t0 = time()
    for i, sample_id in enumerate(indices):
        file_id, shard_sample_id = mapping[sample_id]
        filename = filenames[file_id]
        table = pq.read_table(filename)
        shard_samples = table.to_pylist()
        shard_samples[shard_sample_id]
        times[i] = t = time() - t0
        if progress_bar:
            progress_bar.update(1)
        if time_limit <= t:
            times = times[:i]
            break
    return times


def _clear_mds_files(dataset_root: str) -> None:  # pyright: ignore
    """Clear the intermediate MDS shard files.

    Args:
        dataset_root (str): Dataset root directoyr.
    """
    for parent, _, file_basenames in os.walk(dataset_root):
        for basename in file_basenames:
            if basename.endswith('.mds'):
                filename = os.path.join(parent, basename)
                os.remove(filename)


def _bench_streaming_seq(dataset: StreamingDataset, show_progress: bool,
                         time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in sequential order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        show_progress (bool): Whether to show a progress bar.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    times = np.zeros(dataset.num_samples, np.float64)
    xrange = trange(dataset.num_samples, leave=False) if show_progress else \
        range(dataset.num_samples)
    t0 = time()
    for i in xrange:
        dataset[i]
        times[i] = t = time() - t0
        if time_limit <= t:
            times = times[:i]
            break
    return times


def _bench_streaming_rand(dataset: StreamingDataset, show_progress: bool,
                          time_limit: float) -> NDArray[np.float64]:
    """Benchmark iterating a StreamingDataset in random order.

    Args:
        dataset (StreamingDataset): The streaming dataset to iterate.
        show_progress (bool): Whether to show a progress bar.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        NDArray[np.float64]: Time taken to process that many dataset samples.
    """
    indices = np.random.permutation(dataset.num_samples)
    times = np.zeros(dataset.num_samples)
    if show_progress:
        indices = tqdm(indices, leave=False)
    t0 = time()
    for i, sample_id in enumerate(indices):
        dataset[sample_id]
        times[i] = t = time() - t0
        if time_limit <= t:
            times = times[:i]
            break
    return times


def _to_dict(label: str, times: NDArray[np.float64]) -> Dict[str, Any]:
    """Convert a label and sample latencies ndarray into an interpretable JSON dict.

    Args:
        label (str): Name of this run.
        times (NDArray[np.float64]): Sample access times ndarray.

    Returns:
        Dict[str, Any]: JSON dict of interpretable metadata.
    """
    rate = int(len(times) / times[-1])
    label = f'{label}: {rate:,}/s'
    print(label)
    return {
        'label': label,
        'rate': rate,
        'times': (times * 1e9).astype(np.int64).tolist(),
    }


def _bench_streaming_format(data_root: str, shard_format: str, split: str, show_progress: bool,
                            time_limit: float) -> Dict[str, Any]:
    """Benchmark the performance of a native Stremaing format (e.g., MDS, JSONL, CSV).

    Args:
        data_root (str): Data root directory.
        shard_format (str): Streaming format name.
        split (str): Split name.
        show_progress (bool): Whether to show a progress bar.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        Dict[str, Any]: Mapping of ordering name to benchmark metadata JSON dict.
    """
    dataset_dir = os.path.join(data_root, shard_format, split)
    dataset = StreamingDataset(local=dataset_dir)

    times = _bench_streaming_seq(dataset, show_progress, time_limit)
    seq = _to_dict(f'Streaming {shard_format.upper()} seq', times)

    times = _bench_streaming_rand(dataset, show_progress, time_limit)
    rand = _to_dict(f'Streaming {shard_format.upper()} rand', times)

    return {'seq': seq, 'rand': rand}


def _bench_streaming(data_root: str, split: str, show_progress: bool,
                     time_limit: float) -> Dict[str, Any]:
    """Benchmark the performance of all native Streaming formats.

    Args:
        data_root (str): Data root directory.
        split (str): Split name.
        show_progress (bool): Whether to show a progress bar.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        Dict[str, Any]: Mapping of format to ordering to benchmark metadata JSON dict.
    """
    mds = _bench_streaming_format(data_root, 'mds', split, show_progress, time_limit)
    csv = _bench_streaming_format(data_root, 'csv', split, show_progress, time_limit)
    jsonl = _bench_streaming_format(data_root, 'jsonl', split, show_progress, time_limit)
    return {'mds': mds, 'csv': csv, 'jsonl': jsonl}


def _bench_parquet(data_root: str, split: str, parquet_suffix: str, show_progress: bool,
                   time_limit: float) -> Dict[str, Any]:
    """Benchmark the performance of Parquet and Streaming Parquet.

    Args:
        data_root (str): Data root directory.
        split (str): Split name.
        parquet_suffix (str): Parquet filename suffix.
        show_progress (bool): Whether to show a progress bar.
        time_limit (float): Benchmarking cutoff time.

    Returns:
        Dict[str, Any]: Mapping of benchmark name to ordering to benchmark metadata JSON dict.
    """
    dataset_dir = os.path.join(data_root, 'parquet', split)

    times = _bench_parquet_seq(dataset_dir, parquet_suffix, show_progress, time_limit)
    seq = _to_dict('Parquet seq (in mem)', times)
    times = _bench_parquet_rand(dataset_dir, parquet_suffix, show_progress, time_limit)
    rand = _to_dict('Parquet rand (in mem)', times)
    native = {'seq': seq, 'rand': rand}
    """
    streaming_dataset = StreamingDataset(local=dataset_dir)

    times = _bench_streaming_seq(streaming_dataset, show_progress, time_limit)
    seq = _to_dict('Streaming Parquet seq (cold)', times)
    _clear_mds_files(dataset_dir)
    times = _bench_streaming_rand(streaming_dataset, show_progress, time_limit)
    rand = _to_dict('Streaming Parquet rand (cold)', times)
    cold = {'seq': seq, 'rand': rand}

    times = _bench_streaming_seq(streaming_dataset, show_progress, time_limit)
    seq = _to_dict('Streaming Parquet seq (cached)', times)
    times = _bench_streaming_rand(streaming_dataset, show_progress, time_limit)
    rand = _to_dict('Streaming Parquet rand (cached)', times)
    warm = {'seq': seq, 'rand': rand}
    """

    return {'native': native}


def _bench_lance(data_root: str, split: str, show_progress: bool, time_limit: float,
                 pow_interval: int) -> Dict[str, Any]:
    """Benchmark the performance of Lance and, someday, Streaming Lance.

    Args:
        data_root (str): Data root directory.
        split (str): Split name.
        show_progress (bool): Whether to show a progress bar.
        time_limit (float): Benchmarking cutoff time.
        pow_interval (int): Take count exponent interval. Must be either ``2`` or ``4``.

    Returns:
        Dict[str, Any]: Mapping of take count to ordering to benchmark metadata JSON dict.
    """
    if pow_interval == 4:
        take_counts = 1, 4, 16, 64, 256, 1024
    elif pow_interval == 2:
        take_counts = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    else:
        raise ValueError(f'Unsupported --lance_pow_interval: {pow_interval} (must be 2 or 4).')

    dataset_dir = os.path.join(data_root, 'lance', split)
    lance_dataset = lance.dataset(dataset_dir)

    ret = {}

    for take_count in take_counts:
        times = _bench_lance_seq(lance_dataset, take_count, show_progress, time_limit)
        ret[take_count] = {}
        ret[take_count]['seq'] = _to_dict(f'Lance seq x{take_count:04}', times)

    for take_count in take_counts:
        times = _bench_lance_rand(lance_dataset, take_count, show_progress, time_limit)
        ret[take_count]['rand'] = _to_dict(f'Lance rand x{take_count:04}', times)

    return ret


def main(args: Namespace) -> None:
    """Randomly iterate over a Parquet dataset with Streaming.

    Args:
        args (Namespace): Command-line arguments.
    """
    show_progress = bool(args.progress_bar)

    streaming_info = _bench_streaming(args.data_root, args.split, show_progress, args.time_limit)
    parquet_info = _bench_parquet(args.data_root, args.split, args.parquet_suffix, show_progress,
                                  args.time_limit)
    lance_info = _bench_lance(args.data_root, args.split, show_progress, args.time_limit,
                              args.lance_pow_interval)
    info = {'streaming': streaming_info, 'parquet': parquet_info, 'lance': lance_info}

    with open(args.out, 'w') as out:
        json.dump(info, out)


if __name__ == '__main__':
    main(_parse_args())
