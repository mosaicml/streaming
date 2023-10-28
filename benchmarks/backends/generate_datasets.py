# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a parquet dataset for testing."""

import os
from argparse import ArgumentParser, Namespace
from functools import partial
from shutil import rmtree
from time import time
from typing import List, Optional

import lance
import pyarrow as pa
import pyspark
import pyspark.sql
from delta import configure_spark_with_delta_pip
from pyarrow import parquet as pq
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from task import generate_dataset
from tqdm import tqdm
from wurlitzer import pipes

from streaming import CSVWriter, JSONWriter, MDSWriter


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--show_progress', type=int, default=1)

    args.add_argument('--seed', type=int, default=1337)
    args.add_argument('--num_train', type=int, default=1 << 21)
    args.add_argument('--num_val', type=int, default=1 << 17)

    args.add_argument('--data_root', type=str, default='data/backendss/')
    args.add_argument('--csv', type=str, default='csv')
    args.add_argument('--jsonl', type=str, default='jsonl')
    args.add_argument('--lance', type=str, default='lance')
    args.add_argument('--mds', type=str, default='mds')
    args.add_argument('--parquet', type=str, default='parquet')
    args.add_argument('--delta', type=str, default='delta')

    args.add_argument('--size_limit', type=int, default=1 << 23)
    args.add_argument('--samples_per_shard', type=int, default=1 << 17)
    args.add_argument('--quiet_delta', type=int, default=1)
    return args.parse_args()


def _save_csv(nums: List[int],
              txts: List[str],
              root: str,
              size_limit: Optional[int],
              show_progress: bool = True) -> None:
    """Save the dataset in Streaming CSV form.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        size_limit (int, optional): Maximum shard size in bytes, or no limit.
        show_progress (bool): Whether to show a progress bar while saving. Defaults to ``True``.
    """
    columns = {'num': 'int', 'txt': 'str'}
    with CSVWriter(out=root, columns=columns, size_limit=size_limit) as out:
        each_sample = zip(nums, txts)
        if show_progress:
            each_sample = tqdm(each_sample, total=len(nums), leave=False)
        for num, txt in each_sample:
            sample = {'num': num, 'txt': txt}
            out.write(sample)


def _save_jsonl(nums: List[int],
                txts: List[str],
                root: str,
                size_limit: Optional[int],
                show_progress: bool = True) -> None:
    """Save the dataset Streaming JSONL form.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        size_limit (int, optional): Maximum shard size in bytes, or no limit.
        show_progress (bool): Whether to show a progress bar while saving. Defaults to ``True``.
    """
    columns = {'num': 'int', 'txt': 'str'}
    with JSONWriter(out=root, columns=columns, size_limit=size_limit) as out:
        each_sample = zip(nums, txts)
        if show_progress:
            each_sample = tqdm(each_sample, total=len(nums), leave=False)
        for num, txt in each_sample:
            sample = {'num': num, 'txt': txt}
            out.write(sample)


def _save_mds(nums: List[int],
              txts: List[str],
              root: str,
              size_limit: Optional[int],
              show_progress: bool = True) -> None:
    """Save the dataset in Streaming MDS form.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        size_limit (int, optional): Maximum shard size in bytes, or no limit.
        show_progress (bool): Whether to show a progress bar while saving. Defaults to ``True``.
    """
    columns = {'num': 'int', 'txt': 'str'}
    with MDSWriter(out=root, columns=columns, size_limit=size_limit) as out:
        each_sample = zip(nums, txts)
        if show_progress:
            each_sample = tqdm(each_sample, total=len(nums), leave=False)
        for num, txt in each_sample:
            sample = {'num': num, 'txt': txt}
            out.write(sample)


def _save_parquet(nums: List[int],
                  txts: List[str],
                  root: str,
                  samples_per_shard: int,
                  show_progress: bool = True) -> None:
    """Save the dataset in Streaming MDS form.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        samples_per_shard (int): Maximum numbero of samples per shard.
        show_progress (bool): Whether to show a progress bar while saving. Defaults to ``True``.
    """
    if not os.path.exists(root):
        os.makedirs(root)
    num_samples = len(nums)
    num_shards = (num_samples + samples_per_shard - 1) // samples_per_shard
    each_shard = range(num_shards)
    if show_progress:
        each_shard = tqdm(each_shard, total=num_shards, leave=False)
    for i in each_shard:
        begin = i * samples_per_shard
        end = min(begin + samples_per_shard, num_samples)
        shard_nums = nums[begin:end]
        shard_txts = txts[begin:end]
        path = os.path.join(root, f'{i:05}.parquet')
        obj = {
            'num': shard_nums,
            'txt': shard_txts,
        }
        table = pa.Table.from_pydict(obj)
        pq.write_table(table, path)


def _wrapped_save_delta(nums: List[int], txts: List[str], root: str,
                        samples_per_shard: int) -> None:
    """Save the dataset in Streaming MDS form.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        samples_per_shard (int): Maximum numbero of samples per shard.
    """
    builder = pyspark.sql.SparkSession.builder.appName('deltatorch-example')  # pyright: ignore
    builder = builder.config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension')
    builder = builder.config('spark.sql.catalog.spark_catalog',
                             'org.apache.spark.sql.delta.catalog.DeltaCatalog')
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    schema = StructType([
        StructField('num', IntegerType(), False),
        StructField('txt', StringType(), False),
    ])
    samples = list(zip(nums, txts))
    df = spark.createDataFrame(samples, schema)
    df.write.format('delta').option('maxRecordsPerFile', samples_per_shard).save(root)


def _save_delta(nums: List[int],
                txts: List[str],
                root: str,
                samples_per_shard: int,
                quiet: bool = True) -> None:
    """Save the dataset in Streaming MDS form.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        samples_per_shard (int): Maximum numbero of samples per shard.
        quiet (bool): Whether to capture the Delta logging. Defaults to ``True``.
    """
    bang_on_pipes = lambda: _wrapped_save_delta(nums, txts, root, samples_per_shard)
    if quiet:
        with pipes():
            bang_on_pipes()
    else:
        bang_on_pipes()


def _save_lance(nums: List[int], txts: List[str], root: str, samples_per_shard: int) -> None:
    """Save the dataset in Lance form.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        samples_per_shard (int): Maximum numbero of samples per shard.
    """
    column_names = 'num', 'txt'
    column_values = nums, txts
    table = pa.Table.from_arrays(column_values, column_names)
    lance.write_dataset(table, root, mode='create', max_rows_per_file=samples_per_shard)


def _stat(root: str):
    """Inventory what was written, collecting total files and total bytes.

    Args:
        root (str): Dataset root.

    Returns:
        Tuple[int, int]: Total files and total bytes written.
    """
    rf = 0
    rz = 0
    for p, _, ff in os.walk(root):
        rf += len(ff)
        for f in ff:
            g = os.path.join(p, f)
            rz += os.stat(g).st_size
    return rf, rz


def main(args: Namespace) -> None:
    """Generate identical datasets in various formats for performance comparison.

    Args:
        args (Namespace): Command-line arguments.
    """
    if os.path.exists(args.data_root):
        rmtree(args.data_root)

    kinds = 'csv', 'jsonl', 'lance', 'mds', 'parquet', 'delta'

    show_progress = bool(args.show_progress)
    quiet_delta = bool(args.quiet_delta)

    kind2save = {
        'csv':
            partial(_save_csv, size_limit=args.size_limit, show_progress=show_progress),
        'delta':
            partial(_save_delta, samples_per_shard=args.samples_per_shard, quiet=quiet_delta),
        'jsonl':
            partial(_save_jsonl, size_limit=args.size_limit, show_progress=show_progress),
        'lance':
            partial(_save_lance, samples_per_shard=args.samples_per_shard),
        'mds':
            partial(_save_mds, size_limit=args.size_limit, show_progress=show_progress),
        'parquet':
            partial(_save_parquet,
                    samples_per_shard=args.samples_per_shard,
                    show_progress=show_progress),
    }

    start = time()
    dataset = generate_dataset(args.num_train, args.num_val, show_progress)
    elapsed = time() - start
    print(f'Dataset generation: {elapsed:.3f} sec.')

    for split, nums, txts in dataset:
        print(f'Split {split}:')
        for kind in kinds:
            kind_subdir = getattr(args, kind)
            split_root = os.path.join(args.data_root, 'gold', kind_subdir, split)
            save = kind2save[kind]
            start = time()
            save(nums, txts, split_root)
            elapsed = time() - start
            num_files, num_bytes = _stat(split_root)
            bytes_per_file = num_bytes // num_files
            print(f'* Saving dataset as {kind:8}: {elapsed:8.3f} sec; {num_files:3,} files; ' +
                  f'{num_bytes:12,} bytes; {bytes_per_file:12,} bytes/file.')


if __name__ == '__main__':
    main(parse_args())
