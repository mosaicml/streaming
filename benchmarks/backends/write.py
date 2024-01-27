# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic dataset and serialize it using each Streaming format/backend."""

import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial
from shutil import rmtree
from time import time
from typing import Dict, Iterable, List, Optional, Tuple

import lance
import pyarrow as pa
import pyspark
import pyspark.sql
from delta import configure_spark_with_delta_pip
from pyarrow import parquet as pq
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from tqdm import tqdm
from wurlitzer import pipes

from benchmarks.backends.datagen import generate
from streaming import CSVWriter, JSONLWriter, MDSWriter
from streaming.util.tabulation import Tabulator


def _parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()

    # Reproducibility.
    args.add_argument('--seed', type=int, default=1337)

    # Dataset data distribution.
    args.add_argument('--data_pos_prob', type=float, default=0.75)
    args.add_argument('--data_low', type=int, default=-1_000_000_000)
    args.add_argument('--data_high', type=int, default=1_000_000_000)

    # Sizes of datasets splits and shards.
    args.add_argument('--small', type=int, default=1 << 15)
    args.add_argument('--medium', type=int, default=1 << 20)
    args.add_argument('--large', type=int, default=1 << 25)
    args.add_argument('--size_limit', type=int, default=1 << 23)
    args.add_argument('--samples_per_shard', type=int, default=1 << 18)

    # Outputs.
    args.add_argument('--data_root', type=str, default='data/backends/')
    args.add_argument('--formats', type=str, default='csv,delta,jsonl,lance,mds,parquet')

    # Introspection.
    args.add_argument('--show_progress', type=int, default=1)
    args.add_argument('--quiet_delta', type=int, default=1)

    return args.parse_args()


def _write_csv(nums: List[int],
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
    columns = {
        'num': 'int',
        'txt': 'str',
    }
    with CSVWriter(out=root, columns=columns, size_limit=size_limit) as out:
        each_sample = zip(nums, txts)
        if show_progress:
            each_sample = tqdm(each_sample, total=len(nums), leave=False)
        for num, txt in each_sample:
            sample = {
                'num': num,
                'txt': txt,
            }
            out.write(sample)


def _write_jsonl(nums: List[int],
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
    columns = {
        'num': 'int',
        'txt': 'str',
    }
    with JSONLWriter(out=root, columns=columns, size_limit=size_limit) as out:
        each_sample = zip(nums, txts)
        if show_progress:
            each_sample = tqdm(each_sample, total=len(nums), leave=False)
        for num, txt in each_sample:
            sample = {
                'num': num,
                'txt': txt,
            }
            out.write(sample)


def _write_mds(nums: List[int],
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
    columns = {
        'num': 'int',
        'txt': 'str',
    }
    with MDSWriter(out=root, columns=columns, size_limit=size_limit) as out:
        each_sample = zip(nums, txts)
        if show_progress:
            each_sample = tqdm(each_sample, total=len(nums), leave=False)
        for num, txt in each_sample:
            sample = {
                'num': num,
                'txt': txt,
            }
            out.write(sample)


def _write_parquet(nums: List[int],
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


def _write_delta(nums: List[int], txts: List[str], root: str, samples_per_shard: int) -> None:
    """Save the dataset in Streaming MDS form.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        samples_per_shard (int): Maximum numbero of samples per shard.
    """
    builder = pyspark.sql.SparkSession.builder.appName('prolix')  # pyright: ignore
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


def _do_write_delta(nums: List[int],
                    txts: List[str],
                    root: str,
                    samples_per_shard: int,
                    quietly: bool = True) -> None:
    """Save the dataset in Streaming MDS form, possibly capturing stdout/stderr.

    Args:
        nums (List[int]): The sample numbers.
        txts (List[str]): The sample texts.
        root (str): Root directory.
        samples_per_shard (int): Maximum numbero of samples per shard.
        quietly (bool): Whether to capture the Delta logging. Defaults to ``True``.
    """
    write = lambda: _write_delta(nums, txts, root, samples_per_shard)
    if quietly:
        with pipes():
            write()
    else:
        write()


def _write_lance(nums: List[int], txts: List[str], root: str, samples_per_shard: int) -> None:
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


def _get_file_sizes(root: str) -> List[int]:
    """Inventory what was written, collecting total files and total bytes.

    Args:
        root (str): Dataset root.

    Returns:
        Tuple[int, int]: Total files and total bytes written.
    """
    sizes = []
    for parent, _, file_basenames in sorted(os.walk(root)):
        for basename in sorted(file_basenames):
            path = os.path.join(parent, basename)
            size = os.stat(path).st_size
            sizes.append(size)
    return sizes


def _splits_by_size(dataset: Dict[str, Tuple[List[int], List[str]]]) -> Iterable[str]:
    """Order a dataset's splits by their size in samples, then by name.

    Argxs:
        dataset (Dict[str, Tuple[List[int], List[str]]]): Mapping of split name to split data.

    Returns:
        Iterable[str]: Ordered split names.
    """
    size2splits = defaultdict(list)
    for split, (nums, _) in dataset.items():
        size2splits[len(nums)].append(split)

    splits_by_size = []
    for size in sorted(size2splits):
        for split in sorted(size2splits[size]):
            splits_by_size.append(split)

    return splits_by_size


def main(args: Namespace) -> None:
    """Generate identical datasets in various formats for performance comparison.

    Args:
        args (Namespace): Command-line arguments.
    """
    # Confgure the dataset writing statistics table printer.
    table_columns = '''
        < format 8
        > sec 7
        > samples 12
        > usec/sp 8
        > bytes 14
        > files 6
        > bytes/file 12
        > max bytes/file 14
    '''
    table_indent = 4
    table = Tabulator.from_conf(table_columns, table_indent * ' ')

    # Normalize arguments.
    format_names = args.formats.split(',') if args.formats else []
    show_progress = bool(args.show_progress)
    quiet_delta = bool(args.quiet_delta)

    # Given args, now we know how to configure saving the dataset in each format.
    format2write = {
        'csv':
            partial(_write_csv, size_limit=args.size_limit, show_progress=show_progress),
        'delta':
            partial(_do_write_delta, quietly=quiet_delta,
                    samples_per_shard=args.samples_per_shard),
        'jsonl':
            partial(_write_jsonl, size_limit=args.size_limit, show_progress=show_progress),
        'lance':
            partial(_write_lance, samples_per_shard=args.samples_per_shard),
        'mds':
            partial(_write_mds, size_limit=args.size_limit, show_progress=show_progress),
        'parquet':
            partial(_write_parquet,
                    samples_per_shard=args.samples_per_shard,
                    show_progress=show_progress),
    }

    # Collect sizes of the splits to generate.
    split2size = {
        'small': args.small,
        'medium': args.medium,
        'large': args.large,
    }

    # Generate the dataset samples.
    t0 = time()
    dataset = generate(split2size, args.seed, args.data_pos_prob, args.data_low, args.data_high,
                       show_progress)
    elapsed = time() - t0
    print(f'Generate: {elapsed:.3f} sec.')

    # Wipe output directory if exists.
    if os.path.exists(args.data_root):
        print(f'Found directory at {args.data_root}, wiping it for reuse')
        rmtree(args.data_root)

    # Write each split in each desired formats, in order of size.
    pretty_int = lambda num: f'{num:,}'
    for split in _splits_by_size(dataset):
        print()
        print(f'Write split: {split}')
        print(table.draw_line())
        print(table.draw_header())
        print(table.draw_line())

        nums, txts = dataset[split]
        for format_name in format_names:
            split_root = os.path.join(args.data_root, 'gold', format_name, split)
            write = format2write[format_name]

            t0 = time()
            try:
                write(nums, txts, split_root)
            except:
                continue  # Getting Delta Java OOMs at gigabyte size.
            elapsed = time() - t0

            file_sizes = _get_file_sizes(split_root)
            row = {
                'format': format_name,
                'sec': f'{elapsed:.3f}',
                'samples': pretty_int(len(nums)),
                'usec/sp': f'{1e6 * elapsed / len(nums):.3f}',
                'bytes': pretty_int(sum(file_sizes)),
                'files': pretty_int(len(file_sizes)),
                'bytes/file': pretty_int(sum(file_sizes) // len(file_sizes)),
                'max bytes/file': pretty_int(max(file_sizes)),
            }
            print(table.draw_row(row))
        print(table.draw_line())


if __name__ == '__main__':
    main(_parse_args())
