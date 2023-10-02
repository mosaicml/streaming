# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a Parquet dataset to Lance.

Warning: apparently, Lance will crash with an unhelpful error message if there are any extraneous
files in the Parquet dataset.
"""

from argparse import ArgumentParser, Namespace

import lance
import pyarrow as pa


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--parquet', type=str, required=True)
    args.add_argument('--lance', type=str, required=True)
    return args.parse_args()


def main(args: Namespace) -> None:
    """Convert a Parquet dataset to Lance.

    Args:
        args (Namespace): Command-line arguments.
    """
    dataset = pa.dataset.dataset(args.parquet, format='parquet')
    lance.write_dataset(dataset, args.lance)


if __name__ == '__main__':
    main(parse_args())
