# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Union

import numpy as np
from torch.utils.data import DataLoader

from streaming import MDSWriter, StreamingDataset

_NUM_SAMPLES = 10000
# Word representation of a number
_ONES = ('zero one two three four five six seven eight nine ten eleven twelve '
         'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
_TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()

_COLUMNS = {
    'number': 'int',
    'words': 'str',
}


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:x
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    # compression: Optional[str] = None,
    # hashes: Optional[List[str]] = None,
    # size_limit: Optional[int] = 1 << 26,
    # local: Optional[str] = None,
    # split: Optional[str] = None,
    # download_retry: int = 2,
    # download_timeout: float = 60,
    # validate_hash: Optional[str] = None,
    # keep_zip: bool = False,
    # epoch_size: Optional[int] = None,
    # predownload: Optional[int] = None,
    # cache_limit: Optional[Union[int, str]] = None,
    # partition_algo: str = 'orig',
    # num_canonical_nodes: Optional[int] = None,
    # batch_size: Optional[int] = None,
    # shuffle: bool = False,
    # shuffle_algo: str = 'py1s',
    # shuffle_seed: int = 9176,
    # shuffle_block_size: int = 1 << 18
    args.add_argument(
        '--compression',
        type=str,
        help='Compression or compression:level for MDSWriter.',
    )
    args.add_argument(
        '--hashes',
        type=list[str],
        nargs='+',
        help='List of hash algorithms to apply to shard files for MDSWriter.',
    )
    args.add_argument(
        '--size_limit',
        type=int,
        default=1 << 26,
        help=('Shard size limit, after which point to start a new shard for '
              'MDSWriter. If ``None``, puts everything in one shard.'),
    )
    args.add_argument(
        '--local',
        type=bool,
        default=True,
        help=('Local working directory to download shards to for'
              ' StreamingDataset.'),
    )
    args.add_argument(
        '--split',
        type=str,
        help='Which dataset split to use, if any for StreamingDataset.',
    )
    args.add_argument(
        '--download_retry',
        type=int,
        default=2,
        help=('Number of download re-attempts before giving up for'
              ' StreamingDataset.'),
    )
    args.add_argument(
        '--download_timeout',
        type=float,
        default=60,
        help=('Number of seconds to wait for a shard to download before raising '
              'an exception for StreamingDataset.'),
    )
    args.add_argument(
        '--validate_hash',
        type=str,
        help=('Hash or checksum algorithm to use to validate shards for'
              ' StreamingDataset.'),
    )
    args.add_argument(
        '--keep_zip',
        type=bool,
        default=False,
        help=('Whether to keep or delete the compressed form when decompressing'
              ' downloaded shards for StreamingDataset.'),
    )
    args.add_argument(
        '--epoch_size',
        type=int,
        help=('Number of samples to draw per epoch balanced across all streams'
              ' for StreamingDataset.'),
    )
    args.add_argument(
        '--predownload',
        type=int,
        help=('Target number of samples ahead to download the shards per number'
              ' of workers provided in a dataloader while iterating for'
              ' StreamingDataset.'),
    )
    args.add_argument(
        '--cache_limit',
        type=Union[int, str],
        help="Maximum size in bytes of this StreamingDataset's shard cache.",
    )
    args.add_argument(
        '--partition_algo',
        type=str,
        default='orig',
        help='Which partitioning algorithm to use for StreamingDataset.',
    )
    args.add_argument(
        '--num_canonical_nodes',
        type=int,
        help=('Canonical number of nodes for shuffling with resumption for'
              ' StreamingDataset.'),
    )
    args.add_argument(
        '--batch_size',
        type=int,
        help=('Batch size of its DataLoader, which affects how the dataset is'
              ' partitioned over the workers for StreamingDataset.'),
    )
    args.add_argument(
        '--shuffle',
        type=bool,
        help=('Whether to iterate over the samples in randomized order for'
              ' StreamingDataset.'),
    )
    args.add_argument(
        '--shuffle_algo',
        type=str,
        default='py1s',
        help='Which shuffling algorithm to use for StreamingDataset.',
    )
    args.add_argument(
        '--shuffle_seed',
        type=int,
        default=9176,
        help='Seed for Deterministic data shuffling for StreamingDataset.',
    )
    args.add_argument(
        '--shuffle_block_size',
        type=int,
        default=1 << 18,
        help='Unit of shuffle for StreamingDataset.',
    )
    return args.parse_args()


def say(i: int) -> list[str]:
    """Get the word form of a number.

    Args:
        i (int): The number.

    Returns:
        List[str]: The number in word form.
    """
    if i < 0:
        return ['negative'] + say(-i)
    elif i <= 19:
        return [_ONES[i]]
    elif i < 100:
        return [_TENS[i // 10 - 2]] + ([_ONES[i % 10]] if i % 10 else [])
    elif i < 1_000:
        return [_ONES[i // 100], 'hundred'] + (say(i % 100) if i % 100 else [])
    elif i < 1_000_000:
        return (say(i // 1_000) + ['thousand'] + (say(i % 1_000) if i % 1_000 else []))
    elif i < 1_000_000_000:
        return (say(i // 1_000_000) + ['million'] + (say(i % 1_000_000) if i % 1_000_000 else []))
    else:
        assert False


def get_dataset(num_samples: int) -> list[dict[str, int | str]]:
    """Generate a number-saying dataset of the given size.

    Args:
        num_samples (int): Number of samples.

    Returns:
        list[dict[str, int | str]]: The two generated splits.
    """
    numbers = [((np.random.random() < 0.8) * 2 - 1) * i for i in range(num_samples)]
    samples = []
    for num in numbers:
        words = ' '.join(say(num))
        sample = {'number': num, 'words': words}
        samples.append(sample)
    return samples


# def convert_dataset(
#     out_dir: str,
#     samples: list[dict[str, int | str]],
#     columns: dict[str, str],
#     compression: Optional[str] = None,
#     hashes: Optional[list[str]] = None,
#     size_limit: Optional[int] = 1 << 26,
# ):
#     with MDSWriter(
#         out=out_dir,
#         columns=columns,
#         compression=compression,
#         hashes=hashes,
#         size_limit=size_limit,
#     ) as out:
#         for sample in samples:
#             out.write(sample)

# def load_data(
#     remote: str,
#     local: Optional[str] = None,
#     split: Optional[str] = None,
#     download_retry: int = 2,
#     download_timeout: float = 60,
#     validate_hash: Optional[str] = None,
#     keep_zip: bool = False,
#     epoch_size: Optional[int] = None,
#     predownload: Optional[int] = None,
#     cache_limit: Optional[Union[int, str]] = None,
#     partition_algo: str = 'orig',
#     num_canonical_nodes: Optional[int] = None,
#     batch_size: Optional[int] = None,
#     shuffle: bool = False,
#     shuffle_algo: str = 'py1s',
#     shuffle_seed: int = 9176,
#     shuffle_block_size: int = 1 << 18,
# ):
#     dataset = StreamingDataset(
#         remote=remote,
#         local=local,
#         split=split,
#         download_retry=download_retry,
#         download_timeout=download_timeout,
#         validate_hash=validate_hash,
#         keep_zip=keep_zip,
#         epoch_size=epoch_size,
#         predownload=predownload,
#         cache_limit=cache_limit,
#         partition_algo=partition_algo,
#         num_canonical_nodes=num_canonical_nodes,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         shuffle_algo=shuffle_algo,
#         shuffle_seed=shuffle_seed,
#         shuffle_block_size=shuffle_block_size,
#     )
#     dataloader = DataLoader(dataset)
#     for _ in dataloader:
#         pass


def main(args: Namespace) -> None:
    """Benchmark time taken to generate the epoch for a given dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    dataset = get_dataset(_NUM_SAMPLES)
    with tempfile.TemporaryDirectory() as tmp_upload_dir:
        tmp_dir = tempfile.gettempdir()
        tmp_download_dir = os.path.join(tmp_dir, 'test_regression_basic')
        with MDSWriter(
                out=tmp_upload_dir,
                columns=_COLUMNS,
                compression=args.compression,
                hashes=args.hashes,
                size_limit=args.size_limit,
        ) as out:
            for sample in dataset:
                out.write(sample)

        dataset = StreamingDataset(
            remote=tmp_upload_dir,
            local=tmp_download_dir if args.local else None,
            split=args.split,
            download_retry=args.download_retry,
            download_timeout=args.download_timeout,
            validate_hash=args.validate_hash,
            keep_zip=args.keep_zip,
            epoch_size=args.epoch_size,
            predownload=args.predownload,
            cache_limit=args.cache_limit,
            partition_algo=args.partition_algo,
            num_canonical_nodes=args.num_canonical_nodes,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            shuffle_algo=args.shuffle_algo,
            shuffle_seed=args.shuffle_seed,
            shuffle_block_size=args.shuffle_block_size,
        )
        dataloader = DataLoader(dataset)
        for _ in dataloader:
            pass
    shutil.rmtree(tmp_download_dir, ignore_errors=True)


if __name__ == '__main__':
    main(parse_args())
