# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""C4 (Colossal Cleaned Common Crawl) streaming dataset conversion scripts."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterator

import datasets
from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
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
        '--out_root',
        type=str,
        required=True,
        help='Directory path to store the output dataset',
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
        default=1 << 27,
        help='Shard size limit, after which point to start a new shard. Default: 1 << 27',
    )
    args.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='DataLoader batch size. Default: 512',
    )
    args.add_argument(
        '--progbar',
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


def get(split: str) -> IterableDataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.

    Returns:
        An IterableDataset.
    """

    class ShardedC4(IterableDataset):

        def __init__(self):
            self.dataset = datasets.load_dataset(  # pyright: ignore
                path='c4', name='en', split=split, streaming=True)

        def num_shards(self):
            return len(self.dataset._ex_iterable.kwargs['filepaths'])  # pyright: ignore

        def __iter__(self):
            worker_info = get_worker_info()
            if worker_info:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                shards = self.dataset._ex_iterable.kwargs['filepaths']  # pyright: ignore
                if len(shards) % num_workers:
                    raise ValueError('Shards must divide evenly by num_workers')
                self.dataset._ex_iterable.kwargs['filepaths'] = (  # pyright: ignore
                    shards[worker_id::num_workers])
            return iter(self.dataset)

    return ShardedC4()


def each(dataset: Dataset, num_workers: int, batch_size: int) -> Iterator[Dict[str, Any]]:
    """Iterate over each dataset sample.

    Args:
        dataset (Dataset): A HuggingFace Dataset locally downloaded.
        num_workers (int): DataLoader number of workers.
        batch_size (int): DataLoader batch size.

    Returns:
        Iterator[Dict[str, Any]]: Sample dicts.
    """
    prefetch_factor = max(1, 2 * batch_size // num_workers)
    loader = DataLoader(
        dataset=dataset,  # pyright: ignore
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            yield {key: batch_values[idx] for key, batch_values in batch.items()}


def main(args: Namespace) -> None:
    """Main: create streaming C4 dataset.

    Args:
        args (Namespace): command-line arguments.
    """
    splits = [
        ('train', 'train', 364868892, 64),
        ('validation', 'val', 364608, 8),
    ]
    fields = {'text': 'str', 'timestamp': 'str', 'url': 'str'}
    hashes = get_list_arg(args.hashes)
    for old_split, new_split, num_samples, num_workers in splits:
        dataset = get(old_split)
        split_dir = os.path.join(args.out_root, new_split)
        with MDSWriter(split_dir, fields, args.compression, hashes, args.size_limit) as out:
            samples = each(dataset, num_workers, args.batch_size)  # pyright: ignore
            if args.progbar:
                samples = tqdm(samples, total=num_samples, leave=args.leave)
            for sample in samples:
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
