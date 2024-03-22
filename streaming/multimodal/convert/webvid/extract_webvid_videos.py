# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Convert an MP4-inside MDS dataset to an MP4-outside MDS dataset."""

import os
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from streaming import MDSWriter, StreamingDataset

out_columns = {
    'videoid': 'str',
    'name': 'str',
    'page_idx': 'str',
    'page_dir': 'str',
    'duration': 'int',
    'contentUrl': 'str',
    'content_path': 'str',
}


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--in',
        type=str,
        required=True,
        help='Input mp4-inside MDS dataset directory',
    )
    args.add_argument(
        '--out_mds',
        type=str,
        required=True,
        help='Output mp4-outside MDS dataset directory',
    )
    args.add_argument(
        '--out_mp4',
        type=str,
        required=True,
        help='Output mp4 videos directory',
    )
    args.add_argument(
        '--compression',
        type=str,
        default='zstd:16',
        help='Compression',
    )
    args.add_argument(
        '--hashes',
        type=str,
        default='sha1,xxh3_64',
        help='Hashes',
    )
    args.add_argument(
        '--size_limit',
        type=int,
        default=1 << 25,
        help='Shard size limit',
    )
    return args.parse_args()


def main(args: Namespace) -> None:
    """Convert an MP4-inside MDS dataset to an MP4-outside MDS dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    hashes = args.hashes.split(',') if args.hashes else []
    dataset = StreamingDataset(local=getattr(args, 'in'), batch_size=1)
    with MDSWriter(out=args.out_mds,
                   columns=out_columns,
                   compression=args.compression,
                   hashes=hashes,
                   size_limit=args.size_limit) as out:
        for i, sample in tqdm(enumerate(dataset), total=dataset.num_samples):
            content_path = f'{i // 1000:03}/{i % 1000:03}.mp4'
            filename = os.path.join(args.out_mp4, content_path)
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(filename, 'wb') as out_mp4:
                out_mp4.write(sample['content'])
            del sample['content']
            sample['content_path'] = content_path
            out.write(sample)


if __name__ == '__main__':
    main(parse_args())
