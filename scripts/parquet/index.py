# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Index a parquet dataset for use by Streaming."""

import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterator, Tuple

from pyarrow import parquet as pq


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, required=True)
    args.add_argument('--shard_suffix', type=str, default='.parquet')
    return args.parse_args()


def get_dataset_relative_path(dataset_root: str, path: str) -> str:
    """Get the dataset-relative path of a shard file.

    Args:
        dataset_root (str): Dataset root directory containing this shard.
        path (str): Path to shard under dataset root dir.

    Returns:
        Dataset-relative shard path.
    """
    if not path.startswith(dataset_root):
        raise ValueError('Path {path} was not found under {dataset_root}.')
    rel_path = path[len(dataset_root):]

    while rel_path.startswith(os.path.sep):
        rel_path = rel_path[1:]

    return rel_path


def each_shard_path(dataset_root: str, shard_suffix: str) -> Iterator[Tuple[str, str]]:
    """Collect each Parquet shard, in order.

    Args:
        dataset_root (str): Dataset root directory.
        shard_suffix (str): Suffix of each Parquet shard file.

    Returns:
        Iterator[Tuple[str, str]]: Iterator over absolute and dataset-relative paths.
    """
    for root, _, files in os.walk(dataset_root):
        files = filter(lambda file: file.endswith(shard_suffix), files)
        files = (os.path.join(root, file) for file in files)
        files = sorted(files)
        for path in files:
            dataset_rel_path = get_dataset_relative_path(dataset_root, path)
            yield path, dataset_rel_path


def get_shard_info(path: str, dataset_rel_path: str) -> Dict[str, Any]:
    """Get info the index needs about a Parquet shard.

    Args:
        path (str): Absolute or relative-to-cwd file path.
        dataset_rel_path (str): Relative-to-dataset file path.

    Returns:
        Dict[str, Any]: Shard info.
    """
    num_bytes = os.stat(path).st_size
    table = pq.read_table(path)
    num_samples = len(table)
    return {
        'format': 'parquet',
        'raw_parquet': {
            'basename': dataset_rel_path,
            'bytes': num_bytes,
        },
        'samples': num_samples,
    }


def main(args: Namespace) -> None:
    """Index a parquet dataset for use by Streaming.

    Args:
        args (Namespace): Command-line arguments.
    """
    infos = []
    for path, dataset_rel_path in each_shard_path(args.dataset, args.shard_suffix):
        info = get_shard_info(path, dataset_rel_path)
        infos.append(info)
    obj = {
        'version': 2,
        'shards': infos,
    }
    filename = os.path.join(args.dataset, 'index.json')
    if os.path.exists(filename):
        raise ValueError(f'Index file {filename} already exists.')
    with open(filename, 'w') as out:
        json.dump(obj, out)


if __name__ == '__main__':
    main(parse_args())
