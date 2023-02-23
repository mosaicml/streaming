# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""English Wikipedia 2020-01-01 streaming dataset conversion script."""

import os
from argparse import ArgumentParser, Namespace
from typing import List, Optional

from tqdm import tqdm

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
        help='Local directory path of the input raw dataset',
    )
    args.add_argument(
        '--local',
        type=str,
        required=True,
        help='Local directory path to store the output MDS shard files',
    )
    args.add_argument(
        '--remote',
        type=str,
        help='Remote directory path to upload the output MDS shard files',
    )
    args.add_argument(
        '--compression',
        type=str,
        default='zstd:7',
        help='Compression algorithm to use. Default: zstd:7',
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
        default=1 << 25,
        help='Shard size limit, after which point to start a new shard. Default: 1 << 25',
    )
    args.add_argument(
        '--progress_bar',
        type=int,
        default=1,
        help='tqdm progress bar. Default: 1 (True)',
    )
    args.add_argument(
        '--leave',
        type=int,
        default=0,
        help='Keeps all traces of the progressbar upon termination of iteration. Default: 0 ' +
        '(False)',
    )
    return args.parse_args()


def process_split(in_root: str, local: str, remote: Optional[str], compression: str,
                  hashes: List[str], size_limit: int, progress_bar: int, leave: int,
                  basenames: List[str], split: str) -> None:
    """Process a dataset split.

    Args:
        in_root (str): Local directory path of the input raw dataset.
        local (str): Local directory path to store the output MDS shard files.
        remote(str, optional): Remote directory path to upload the output MDS shard files.
        compression (str): Which compression to use, or empty string if none.
        hashes (List[str]): List of hashes to store of the shards.
        size_limit (int): Maximum shard size in bytes.
        progress_bar (int): Whether to display a progress bar.
        leave (int): Whether to leave the progress bar.
        basenames (List[str]): List of input shard basenames.
        split (str): Split name.
    """
    local_split_dir = os.path.join(local, split)
    remote_split_dir = None
    if remote:
        remote_split_dir = os.path.join(remote, split)
    columns = {'text': 'str'}
    with MDSWriter(local=local_split_dir,
                   remote=remote_split_dir,
                   columns=columns,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit,
                   progress_bar=progress_bar) as out:
        if progress_bar:
            basenames = tqdm(basenames, leave=leave)
        for basename in basenames:
            filename = os.path.join(in_root, basename)
            for line in open(filename):
                line = line.strip()
                if not line:
                    continue
                sample = {'text': line}
                out.write(sample)


def main(args: Namespace) -> None:
    """Main: create streaming enwiki dataset.

    Args:
        args (Namespace): command-line arguments.
    """
    hashes = get_list_arg(args.hashes)

    basenames = [f'part-{i:05}-of-00500' for i in range(500)]
    split = 'train'
    process_split(args.in_root, args.local, args.remote, args.compression, hashes, args.size_limit,
                  args.progress_bar, args.leave, basenames, split)

    basenames = ['eval.txt']
    split = 'val'
    process_split(args.in_root, args.local, args.remote, args.compression, hashes, args.size_limit,
                  args.progress_bar, args.leave, basenames, split)


if __name__ == '__main__':
    main(parse_args())
