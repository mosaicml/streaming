from argparse import ArgumentParser, Namespace
import datasets
from datasets import Dataset  # pyright: ignore
import os
from torch.utils.data import DataLoader, get_worker_info, IterableDataset
from tqdm import tqdm
from typing import Any, Dict, Iterator

from ...base import MDSWriter


def parse_args() -> Namespace:
    """Parse commandline arguments.
    Args:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--out', type=str, default='/datasets/mds/c4-zstd/')
    args.add_argument('--compression', type=str, default='zstd:7')
    args.add_argument('--hashes', type=str, default='sha1,xxh64')
    args.add_argument('--limit', type=int, default=1 << 27)
    args.add_argument('--batch_size', type=int, default=512)
    args.add_argument('--progbar', type=int, default=1)
    args.add_argument('--leave', type=int, default=0)
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
            self.dataset = datasets.load_dataset(path='c4', name='en', split=split,  # pyright: ignore
                                                 streaming=True)  # pyright: ignore

        def num_shards(self):
            return len(self.dataset._ex_iterable.kwargs['filepaths'])

        def __iter__(self):
            worker_info = get_worker_info()
            if worker_info:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                shards = self.dataset._ex_iterable.kwargs['filepaths']
                assert len(shards) % num_workers == 0
                self.dataset._ex_iterable.kwargs['filepaths'] = shards[worker_id::num_workers]
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
    loader = DataLoader(dataset=dataset,  # pyright: ignore
                        batch_size=batch_size,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor)
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            yield {key: batch_values[idx] for key, batch_values in batch.items()}


def main(args: Namespace) -> None:
    """Main: create streaming CIFAR10 dataset.
    Args:
        args (Namespace): Commandline arguments.
    """
    splits = [
        ('train', 'train', 364868892, 64),
        ('validation', 'val', 364608, 8),
    ]
    fields = {
        'text': 'str',
        'timestamp': 'str',
        'url': 'str'
    }
    hashes = args.hashes.split(',') if args.hashes else []
    for old_split, new_split, num_samples, num_workers in splits:
        dataset = get(old_split)
        split_dir = os.path.join(args.out, new_split)
        with MDSWriter(split_dir, fields, args.compression, hashes, args.limit) as out:
            samples = each(dataset, num_workers, args.batch_size)  # pyright: ignore
            if args.progbar:
                samples = tqdm(samples, total=num_samples, leave=args.leave)
            for sample in samples:
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
