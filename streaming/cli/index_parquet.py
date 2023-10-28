# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a Streaming index file for the given Parquet dataset."""

import json
from argparse import ArgumentParser, Namespace

from streaming.format import index_parquet
from streaming.util.pretty import unpack_str2str


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--local', type=str, required=True, help='Path to dataset cache.')
    args.add_argument('--remote', type=str, default='', help='Path to gold copy of dataset.')
    args.add_argument('--split', type=str, default='', help='Dataset split subdir.')
    args.add_argument('--keep', type=str, default='', help='Optional regex for filtering shards.')
    args.add_argument('--num_procs',
                      type=int,
                      default=0,
                      help='Process parallelism. Set to -1 for single process, 0 for <number ' +
                      'of CPUs> processes, and positive int for that many processes.')
    args.add_argument('--download_timeout',
                      type=str,
                      default='2m',
                      help='Download timeout per Parquet file.')
    args.add_argument('--max_file_bytes',
                      type=str,
                      default='200m',
                      help='Maximum file size in bytes, or 0 to disable..')
    args.add_argument('--same_schema',
                      type=int,
                      default=1,
                      help='Whether all shards must be of the same MDS schema.')
    args.add_argument('--columns',
                      type=str,
                      default='',
                      help='Override hte inferred schema to set any field names and types ' +
                      'specified here.')
    args.add_argument('--show_progress', type=int, default=1, help='Show progress bar.')
    args.add_argument('--sort_keys', type=int, default=1, help='Whether to sort JSON keys.')
    args.add_argument('--indent', type=int, default=-1, help='JSON indent level (0 to disable).')
    return args.parse_args()


def main(args: Namespace) -> None:
    """Generate a Streaming index for the given Parquet dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    columns = unpack_str2str(args.columns)
    obj = index_parquet(local=args.local,
                        remote=args.remote,
                        split=args.split,
                        keep=args.keep,
                        num_procs=args.num_procs,
                        download_timeout=args.download_timeout,
                        max_file_bytes=args.max_file_bytes,
                        same_schema=args.same_schema,
                        columns=columns,
                        show_progress=args.show_progress)

    indent = None if args.indent < 0 else args.indent
    text = json.dumps(obj, sort_keys=args.sort_keys, indent=indent)
    print(text)


if __name__ == '__main__':
    main(parse_args())
