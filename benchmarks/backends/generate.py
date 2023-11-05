# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate copies of the same dataset in different Streaming formats."""
import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial
from shutil import rmtree
from time import time
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

import lance
import numpy as np
import pyarrow as pa
import pyspark
import pyspark.sql
from delta import configure_spark_with_delta_pip
from numpy.random import Generator
from pyarrow import parquet as pq
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from tqdm import tqdm
from wurlitzer import pipes

from streaming import CSVWriter, JSONWriter, MDSWriter
from streaming.util.tabulator import Tabulator


def _parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()

    # Reproducibility.
    args.add_argument('--seed', type=int, default=1337)

    # Dataset properties.
    args.add_argument('--data_pos_prob', type=float, default=0.75)
    args.add_argument('--data_low', type=int, default=-1_000_000_000)
    args.add_argument('--data_high', type=int, default=1_000_000_000)

    # Sizes of dataset splits and shards.
    args.add_argument('--small', type=int, default=1 << 15)
    args.add_argument('--medium', type=int, default=1 << 20)
    args.add_argument('--large', type=int, default=1 << 25)
    args.add_argument('--size_limit', type=int, default=1 << 23)
    args.add_argument('--samples_per_shard', type=int, default=1 << 18)

    # Output root.
    args.add_argument('--data_root', type=str, default='data/backends/')

    # Formats to output.
    args.add_argument('--formats', type=str, default='csv,delta,jsonl,lance,mds,parquet')

    # Logging.
    args.add_argument('--show_progress', type=int, default=1)
    args.add_argument('--quiet_delta', type=int, default=1)

    return args.parse_args()


def _generate_int(rng: Generator,
                  pos_prob: float = 0.75,
                  low: int = -1_000_000_000,
                  high: int = 1_000_000_000) -> int:
    """Pick a random integer to say in words.

    This is a synthetic dataset whose random numbers need to be distinct, deterministic given a
    seed, and little else. We choose a distribution that seems the most pleasing to us.

    Properties:
      * About 80% positive and 20% negative.
      * Magnitude of up to a billion on either side of zero.
      * Strongly skewed toward the origin, i.e. chosen uniformly across base-10 digit lengths (at
        least until running out of integers of that length anyway).

    Args:
        rng (Generator): NumPy random number generator.
        pos_prob (float): Probability of output being positive. Defaults to ``0.75``.
        low (int): Minimum of output range. Must be negative. Defaults to ``-1_000_000_000``.
        high (int): Maximum of output range. Must be positive. Defaults to ``1_000_000_000``.
    """
    if not 0 <= pos_prob <= 1:
        raise ValueError(f'Invalid positive probability ``pos_prob``: 0 <= {pos_prob} <= 1.')

    if not low < 0 < high:
        raise ValueError(f'Invalid sampling range ``low`` and/or ``high``: {low} < 0 < {high}.')

    is_pos = rng.uniform() < pos_prob
    max_digits = np.log10(high) if is_pos else np.log10(-low)
    exponent = rng.uniform(0, max_digits)
    magnitude = int(10**exponent)
    sign = is_pos * 2 - 1
    return sign * magnitude


def _generate_ints(count: int,
                   seed: int = 0x1337,
                   pos_prob: float = 0.75,
                   low: int = -1_000_000_000,
                   high: int = 1_000_000_000,
                   show_progress: bool = True) -> List[int]:
    """Sample until we have the given number of distinct integers.

    Args:
        count (int): How many samples to draw.
        seed (int): Seed for the random number generator. Defaults to ``0x1337``.
        pos_prob (float): Probability of output being positive. Defaults to ``0.75``.
        low (int): Minimum of output range. Must be negative. Defaults to ``-1_000_000_000``.
        high (int): Maximum of output range. Must be positive. Defaults to ``1_000_000_000``.
        show_progress (bool): Whether to display a progress bar. Defaults to ``True``.

    Returns:
        List[int]: The integers that were drawn.
    """
    rng = np.random.default_rng(seed)
    nums = set()
    progress_bar = tqdm(total=count, leave=False) if show_progress else None
    while len(nums) < count:
        num = _generate_int(rng)
        if num in nums:
            continue

        nums.add(num)
        if progress_bar:
            progress_bar.update(1)
    if progress_bar:
        progress_bar.close()

    nums = sorted(nums)
    rng.shuffle(nums)
    return nums


_ones = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
         'fifteen sixteen seventeen eighteen nineteen').split()

_tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def _int_to_words(num: int) -> List[str]:
    """Say an integer as a list of words.

    Args:
        num (int): The integer.

    Returns:
        List[str]: The integer as a list of words.
    """
    if num < 0:
        return ['negative'] + _int_to_words(-num)
    elif num <= 19:
        return [_ones[num]]
    elif num < 100:
        tens = [_tens[num // 10 - 2]]
        ones = [_ones[num % 10]] if num % 10 else []
        return tens + ones
    elif num < 1_000:
        hundreds = [_ones[num // 100], 'hundred']
        etc = _int_to_words(num % 100) if num % 100 else []
        return hundreds + etc
    elif num < 1_000_000:
        thousands = _int_to_words(num // 1_000) + ['thousand']
        etc = _int_to_words(num % 1_000) if num % 1_000 else []
        return thousands + etc
    elif num < 1_000_000_000:
        millions = _int_to_words(num // 1_000_000) + ['million']
        etc = _int_to_words(num % 1_000_000) if num % 1_000_000 else []
        return millions + etc
    else:
        raise ValueError('Integer out of range: -1,000,000,000 < {num} < +1,000,000,000.')


def _int_to_text(num: int) -> str:
    """Say an integer as text.

    Args:
        num (int): The integer.

    Returns:
        str: The integer as text.
    """
    words = _int_to_words(num)
    return ' '.join(words)


T = TypeVar('T')


def _split(items: List[T], sizes: List[int]) -> List[List[T]]:
    """Divide the given items across the splits given by their sizes.

    Args:
        items (List[Any]): The items to divide across the spans.
        sizes (List[int]): Number of items per split.

    Returns:
        List[List[Any]]: Each split of items.
    """
    total = sum(sizes)
    if len(items) != total:
        raise ValueError(f'Number of items must match the combined size of the splits: ' +
                         f'{len(items)} items vs splits of size {sizes} = {total}.')

    splits = []
    begin = 0
    for size in sizes:
        split = items[begin:begin + size]
        splits.append(split)
        begin += size

    return splits


def _generate_dataset(split2size: Dict[str, int],
                      seed: int = 0x1337,
                      pos_prob: float = 0.75,
                      low: int = -1_000_000_000,
                      high: int = 1_000_000_000,
                      show_progress: bool = True) -> Dict[str, Tuple[List[int], List[str]]]:
    """Generate a dataset, made of splits, to be saved in different forms for comparison.

    Args:
        split2size (Dict[str, int]): Mapping of split name to size in samples.
        seed (int): Seed for the random number generator. Defaults to ``0x1337``.
        pos_prob (float): Probability of output being positive. Defaults to ``0.75``.
        low (int): Minimum of output range. Must be negative. Defaults to ``-1_000_000_000``.
        high (int): Maximum of output range. Must be positive. Defaults to ``1_000_000_000``.
        show_progress (bool): Whether to show a progress bar. Defaults to ``True``.

    Returns:
        Dict[str, Tuple[List[int], List[str]]]: Mapping of split name to nums and texts.
    """
    split_sizes = []
    total = 0
    for split in sorted(split2size):
        size = split2size[split]
        split_sizes.append(size)
        total += size

    nums = _generate_ints(total, seed, low, high, show_progress)
    nums_per_split = _split(nums, split_sizes)

    texts = list(map(_int_to_text, nums))
    texts_per_split = _split(texts, split_sizes)

    dataset = {}
    for index, split in enumerate(sorted(split2size)):
        dataset[split] = nums_per_split[index], texts_per_split[index]

    return dataset


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
    with JSONWriter(out=root, columns=columns, size_limit=size_limit) as out:
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
        > sec 6
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
    dataset = _generate_dataset(split2size, args.seed, args.data_pos_prob, args.data_low,
                                args.data_high, show_progress)
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
