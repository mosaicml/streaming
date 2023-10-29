# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a Streaming index file for the given Parquet dataset."""

from argparse import ArgumentParser, Namespace

from streaming.hashing import get_hash, get_hashes
from streaming.util.pretty import unpack_strs


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    supported = sorted(get_hashes())
    args = ArgumentParser()
    args.add_argument('--file', type=str, required=True, help='Path to file to hash.')
    args.add_argument('--hash',
                      type=str,
                      required=True,
                      help=f'Comma-delimted names of hash algortihms. Must be in this list: ' +
                      f'{supported}. Names and hex digests will be listed one per line.')
    return args.parse_args()


def main(args: Namespace):
    """Calculate one or more hashes of the data of the given file.

    Args:
        args (Namespace): Command-line arguments.
    """
    data = open(args.file, 'rb').read()
    for algo in unpack_strs(args.hash):
        hex_digest = get_hash(algo, data)
        print(f'{algo} {hex_digest}')


if __name__ == '__main__':
    main(parse_args())
