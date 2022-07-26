from argparse import ArgumentParser, Namespace
import datasets
from datasets import Dataset  # pyright: ignore
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, Iterator

from ...base.mds.writer import MDSCreator


def parse_args() -> Namespace:
    """Parse commandline arguments.

    Args:
        Namespace: Commandline arguments.
    """
    args = ArgumentParser()
    args.add_argument('--out', type=str, default='/datasets/mds/c4/')
    args.add_argument('--compression', type=str, default='')
    args.add_argument('--hashes', type=str, default='sha1,xxh64')
    args.add_argument('--limit', type=int, default=1 << 27)
    args.add_argument('--num_workers', type=int, default=64)
    args.add_argument('--batch_size', type=int, default=512)
    args.add_argument('--progbar', type=int, default=1)
    args.add_argument('--leave', type=int, default=0)
    return args.parse_args()


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
        ('train', 'train', 364868892),
        ('validation', 'val', 364608),
    ]
    fields = {
        'text': 'str',
        'timestamp': 'str',
        'url': 'str'
    }
    hashes = args.hashes.split(',') if args.hashes else []
    for old_split, new_split, num_samples in splits:
        dataset = datasets.load_dataset(path='c4', name='en', split=old_split)  # pyright: ignore
        split_dir = os.path.join(args.out, new_split)
        with MDSCreator(split_dir, fields, args.compression, hashes, args.limit) as out:
            samples = each(dataset, args.num_workers, args.batch_size)  # pyright: ignore
            if args.progbar:
                samples = tqdm(samples, total=num_samples, leave=args.leave)
            for sample in samples:
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
