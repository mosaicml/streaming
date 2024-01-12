# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Post-process a set of subdirectories containing shards into one unified dataset."""

from argparse import ArgumentParser, Namespace
from glob import glob
import json
import os
import shutil


def parse_args() -> Namespace:
    """Parse commmand-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True,
                      help='Location of input shard directories named like group-###')
    args.add_argument('--out_root', type=str, required=True,
                      help='Location of merged shards (a valid streaming dataset)')
    return args.parse_args()


def main(args: Namespace) -> None:
    """Post-process a set of shard subdirectories into one unified dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    os.makedirs(args.out_root)
    pattern = os.path.join(args.in_root, 'group-*')
    subdirs = sorted(glob(pattern))
    offset = 0
    infos = []
    for subdir in subdirs:
        shards_this_group = len(os.listdir(subdir)) - 1

        # Move shard files.
        for shard in range(shards_this_group):
            old_filename = os.path.join(subdir, f'shard.{shard:05d}.mds.zstd')
            new_filename = os.path.join(args.out_root, f'shard.{offset + shard:05d}.mds.zstd')
            os.rename(old_filename, new_filename)

        # Collect shard infos.
        index_filename = os.path.join(subdir, 'index.json')
        obj = json.load(open(index_filename))
        infos += obj['shards']

        # Update offset.
        offset += shards_this_group

    # Update the indices of the collected shard infos to be global.
    for shard, info in enumerate(infos):
        info['raw_data']['basename'] = f'shard.{shard:05d}.mds'
        info['zip_data']['basename'] = f'shard.{shard:05d}.mds.zstd'

    # Create new index.
    obj = {
        'version': 2,
        'shards': infos,
    }
    index_filename = os.path.join(args.out_root, 'index.json')
    with open(index_filename, 'w') as out:
        json.dump(obj, out)

    # Remove leftover old index files.
    shutil.rmtree(args.in_root)


if __name__ == '__main__':
    main(parse_args())
