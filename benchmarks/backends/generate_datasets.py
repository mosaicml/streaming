# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a parquet dataset for testing."""

import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial
from shutil import rmtree
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lance
import pyarrow as pa
import pyspark
import pyspark.sql
from delta import configure_spark_with_delta_pip
from pyarrow import parquet as pq
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from task import generate
from tqdm import tqdm
from typing_extensions import Self
from wurlitzer import pipes

from streaming import CSVWriter, JSONWriter, MDSWriter


def _parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()

    # Reproducibility.
    args.add_argument('--seed', type=int, default=1337)

    # Dataset and shard sizes.
    args.add_argument('--small', type=int, default=1 << 16)
    args.add_argument('--medium', type=int, default=1 << 20)
    args.add_argument('--large', type=int, default=1 << 24)
    args.add_argument('--size_limit', type=int, default=1 << 23)
    args.add_argument('--samples_per_shard', type=int, default=1 << 18)

    # Output root.
    args.add_argument('--data_root', type=str, default='data/backends/')

    # Formats to output.
    args.add_argument('--formats', type=str, default='csv,delta,jsonl,lance,mds,parquet')

    # Output subdir per format.
    args.add_argument('--csv', type=str, default='csv')
    args.add_argument('--jsonl', type=str, default='jsonl')
    args.add_argument('--lance', type=str, default='lance')
    args.add_argument('--mds', type=str, default='mds')
    args.add_argument('--parquet', type=str, default='parquet')
    args.add_argument('--delta', type=str, default='delta')

    # Logging.
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


class Tabulator:
    """Line by line text table printer.

    Example:
        conf = '''
            < format 8
            > sec 6
            > samples 12
            > usec/sp 8
            > bytes 14
            > files 6
            > bytes/file 12
            > max bytes/file 14
        '''
        left = 4 * ' '
        tab = Tabulator.from_conf(conf, left)

    Args:
        cols (List[Tuple[str, str, int]]: Each column config (i.e., just, name, width).
        left (str, optional): Print this before each line (e.g., indenting). Defaults to ``None``.
    """

    def __init__(self, cols: List[Tuple[str, str, int]], left: Optional[str] = None) -> None:
        self.cols = cols
        self.col_justs = []
        self.col_names = []
        self.col_widths = []
        for just, name, width in cols:
            if just not in {'<', '>'}:
                raise ValueError(f'Invalid justify (must be one of "<" or ">"): {just}.')

            if not name:
                raise ValueError('Name must be non-empty.')
            elif width < len(name):
                raise ValueError(f'Name is too wide for its column width: {width} vs {name}.')

            if width <= 0:
                raise ValueError(f'Width must be positive, but got: {width}.')

            self.col_justs.append(just)
            self.col_names.append(name)
            self.col_widths.append(width)

        self.left = left

        self.box_chr_horiz = chr(0x2500)
        self.box_chr_vert = chr(0x2502)

    @classmethod
    def from_conf(cls, conf: str, left: Optional[str] = None) -> Self:
        """Initialize a Tabulator from a text table defining its columns.

        Args:
            conf (str): The table config.
            left (str, optional): Optional string that is printed before each line (e.g., indents).
        """
        cols = []
        for line in conf.strip().split('\n'):
            words = line.split()

            if len(words) < 3:
                raise ValueError(f'Invalid col config (must be "just name width"): {line}.')

            just = words[0]
            name = ' '.join(words[1:-1])
            width = int(words[-1])
            cols.append((just, name, width))
        return cls(cols, left)

    def draw_row(self, row: Dict[str, Any]) -> str:
        """Draw a row, given a mapping of column name to field value.

        Args:
            row (Dict[str, Any]): Mapping of column name to field value.

        Returns:
            str: Text line.
        """
        fields = []
        for just, name, width in self.cols:
            val = row[name]

            txt = val if isinstance(val, str) else str(val)
            if width < len(txt):
                raise ValueError(f'Field is too wide for its column: column (just: {just}, ' +
                                 f'name: {name}, width: {width}) vs field {txt}.')

            txt = txt.ljust(width) if just == '<' else txt.rjust(width)
            fields.append(txt)

        left_txt = self.left or ''
        fields_txt = f' {self.box_chr_vert} '.join(fields)
        return f'{left_txt}{self.box_chr_vert} {fields_txt} {self.box_chr_vert}'

    def draw_header(self) -> str:
        """Draw a header row.

        Returns:
            str: Text line.
        """
        row = dict(zip(self.col_names, self.col_names))
        return self.draw_row(row)

    def draw_line(self) -> str:
        """Draw a divider row.

        Returns:
            str: Text line.
        """
        seps = (self.box_chr_horiz * width for width in self.col_widths)
        row = dict(zip(self.col_names, seps))
        line = self.draw_row(row)
        return line.replace(self.box_chr_vert, self.box_chr_horiz)


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
    # Normalize arguments.
    format_names = args.formats.split(',') if args.formats else []
    show_progress = bool(args.show_progress)
    quiet_delta = bool(args.quiet_delta)
    split2size = {
        'small': args.small,
        'medium': args.medium,
        'large': args.large,
    }

    # Wipe output directory if exists.
    if os.path.exists(args.data_root):
        rmtree(args.data_root)

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

    # Now, generate the dataset.
    t0 = time()
    dataset = generate(split2size, show_progress)
    elapsed = time() - t0
    print(f'Generate: {elapsed:.3f} sec.')

    # Confgure the text table printer for dataset writing info.
    conf = '''
        < format 8
        > sec 6
        > samples 12
        > usec/sp 8
        > bytes 14
        > files 6
        > bytes/file 12
        > max bytes/file 14
    '''
    left = 4 * ' '
    tab = Tabulator.from_conf(conf, left)

    # Write each split in each desired formats, in order of size.
    pretty_int = lambda num: f'{num:,}'
    for split in _splits_by_size(dataset):
        print()
        print(f'Write split: {split}')
        print(tab.draw_line())
        print(tab.draw_header())
        print(tab.draw_line())
        nums, txts = dataset[split]
        for format_name in format_names:
            format_subdir = getattr(args, format_name)
            split_root = os.path.join(args.data_root, 'gold', format_subdir, split)
            write = format2write[format_name]

            t0 = time()
            write(nums, txts, split_root)
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
            print(tab.draw_row(row))
        print(tab.draw_line())


if __name__ == '__main__':
    main(_parse_args())
