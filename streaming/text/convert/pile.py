# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the Pile dataset to streaming format."""

import json
import os
from argparse import ArgumentParser, Namespace
from collections import Counter
from glob import glob
from multiprocessing import Pool
from typing import Dict, Iterator, List, Tuple

from streaming.base import MDSWriter
from streaming.base.util import get_list_arg


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Args:
        Namespace: command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--in_root',
        type=str,
        required=True,
        help='Directory path to store the input dataset',
    )
    args.add_argument(
        '--out_root',
        type=str,
        required=True,
        help='Directory path to store the output dataset',
    )
    args.add_argument(
        '--compression',
        type=str,
        default='zst:16',
        help='Compression algorithm to use. Default: zst:16',
    )
    args.add_argument(
        '--hashes',
        type=str,
        default='sha1,xxh64',
        help='Hashing algorithms to apply to shard files. Default: sha1,xxh64',
    )
    args.add_argument(
        '--size_limit',
        type=int,
        default=1 << 27,
        help='Shard size limit, after which point to start a new shard. Default: 1 << 27',
    )
    return args.parse_args()


def each_task(in_root: str, out_root: str, compression: str, hashes: List[str], size_limit: int,
              in_files: List[str]) -> Iterator[Tuple[str, str, str, List[str], int]]:
    """Get the arg tuple corresponding to each JSONL input file to convert to streaming.

    Args:
        in_root (str): Root directory of input JSONL files.
        out_root (str): Root directory of output MDS files.
        compression (str): Which compression algorithm to use, or empty if none.
        hashes (List[str]): Hashing algorithms to apply to shard files.
        size_limit (int): Shard size limit, after which point to start a new shard.
        in_files (List[str]): List of input files to generate arguments for.

    Returns:
        Iterator[Tuple[str, str, str, List[str], int]]: Each argument tuple.
    """
    for in_file in in_files:
        assert in_file.startswith(in_root)
        assert in_file.endswith('.jsonl')
        out_dir = os.path.join(out_root, in_file[len(in_root):-len('.jsonl')])
        yield in_file, out_dir, compression, hashes, size_limit


def file_to_dir(args: Tuple[str, str, str, List[str], int]) -> Dict[str, int]:
    """Convert a JSONL input file into a directory of MDS shards.

    This is the unit of work executed by the process pool.

    Args:
        args (Tuple[str, str, str, List[str], int): All arguments, packed into a tuple because
            process pools only pass one argument.

    Returns:
        Dict[str, int]: Count of how many samples belonged to each Pile dataset subset.
    """
    in_file, out_dir, compression, hashes, size_limit = args

    columns = {
        'text': 'str',
        'pile_set_name': 'str',
    }

    counts = Counter()
    with MDSWriter(out_dir, columns, compression, hashes, size_limit) as out:
        for line in open(in_file):
            obj = json.loads(line)
            if sorted(obj.keys()) != ['meta', 'text']:
                raise ValueError('Invalid sample fields.')
            text = obj['text']
            meta = obj['meta']
            if sorted(meta.keys()) != ['pile_set_name']:
                raise ValueError('Invalild sample meta fields.')
            pile_set_name = meta['pile_set_name']
            sample = {
                'text': text,
                'pile_set_name': pile_set_name,
            }
            out.write(sample)
            counts[pile_set_name] += 1
    return counts


def main(args: Namespace) -> None:
    """Convert the Pile to streaming format.

    Args:
        args (Namespace): Command-line arguments.
    """
    hashes = get_list_arg(args.hashes)

    # Find the original JSONL files to convert.
    pattern = os.path.join(args.in_root, 'train', '*.jsonl')
    trains = sorted(glob(pattern))
    val = os.path.join(args.in_root, 'val.jsonl')
    test = os.path.join(args.in_root, 'test.jsonl')
    in_files = trains + [val, test]

    # Get the arguments for each JSONL file conversion.
    arg_tuples = each_task(args.in_root, args.out_root, args.compression, hashes, args.size_limit,
                           in_files)

    # Process each JSONL file in parallel into directories of shards.
    with Pool() as pool:
        counters = pool.imap(file_to_dir, arg_tuples)
        for in_file, counts in zip(in_files, counters):
            obj = {
                'file': in_file,
                'counts': counts,
            }
            print(json.dumps(obj, sort_keys=True))


if __name__ == '__main__':
    main(parse_args())
