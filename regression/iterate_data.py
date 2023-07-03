# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create a streaming dataset from toy data with various options for regression testing."""

import os
import shutil
import tempfile
from argparse import ArgumentParser, Namespace

import utils
from torch.utils.data import DataLoader

from streaming import StreamingDataset
from streaming.base.distributed import barrier

_TRAIN_EPOCHS = 2


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:x
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--cloud', type=str)
    args.add_argument('--local', default=False, action='store_true')
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
        default=False,
        action='store_true',
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
        type=str,
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
        default=False,
        action='store_true',
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


def main(args: Namespace) -> None:
    """Benchmark time taken to generate the epoch for a given dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    tmp_dir = tempfile.gettempdir()
    tmp_download_dir = os.path.join(tmp_dir, 'test_regression_download')
    dataset = StreamingDataset(
        remote=utils.get_upload_dir(args.cloud),
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
    for _ in range(_TRAIN_EPOCHS):
        for _ in dataloader:
            pass

    barrier()
    # Clean up directories
    for stream in dataset.streams:
        shutil.rmtree(stream.local, ignore_errors=True)
    shutil.rmtree(tmp_download_dir, ignore_errors=True)


if __name__ == '__main__':
    main(parse_args())
