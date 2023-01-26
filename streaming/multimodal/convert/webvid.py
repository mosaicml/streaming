# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Download videos, creating an MDS dataset.

Instructions:

1. Navigate to the download section of https://m-bain.github.io/webvid-dataset/, where you will
   find 2.5M and 10M dataset splits:

   2.5M:
   - train: http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv (640MB)
   - val: http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_val.csv (1.3MB)

   10M:
   - train: http://www.robots.ox.ac.uk/~maxbain/webvid/results_10M_train.csv (2.7GB)
   - val: http://www.robots.ox.ac.uk/~maxbain/webvid/results_10M_val.csv (1.3MB)

2. Download each CSV you want to process.

3. Run this script with flags --in (CSV) --out (MDS dir)
"""

import csv
import re
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from typing import Any, Dict, Iterator, Optional

import requests

from streaming import MDSWriter

# For parsing the duration field.
duration_pattern = re.compile('^PT\\d{2}H\\d{2}M\\d{2}S$')
digits_pattern = re.compile('\\d{2}')


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in',
                      type=str,
                      required=True,
                      help='Dataset CSV file from https://m-bain.github.io/webvid-dataset/')
    args.add_argument('--out',
                      type=str,
                      required=True,
                      help='Path to MDS dataset that we will create')
    args.add_argument('--num_procs',
                      type=int,
                      default=64,
                      help='Number of processes to use for downloading videos')
    args.add_argument(
        '--limit',
        type=int,
        default=-1,
        help='Only process the first "limit" number of samples, or all of them if set to -1')
    return args.parse_args()


def each_todo(filename: str) -> Iterator[Dict[str, Any]]:
    """Get each sample to download.

    Args:
        filename (str): Path to CSV file containing samples to download.

    Returns:
        Iterator[Dict[str, Any]]: Each sample to download.
    """
    it = csv.reader(open(filename))
    header = next(it)
    for row in it:
        yield dict(zip(header, row))


def head(items: Iterator, limit: int) -> Iterator:
    """Take the first "limit" number of items from an iterator.

    Args:
        items (Iterator): The iterator over the items.
        limit (int): Maximum number of items to return.

    Returns:
        Iterator: An iterator over the first "limit" items.
    """
    for i, item in enumerate(items):
        if i == limit:
            return
        yield item


def parse_duration(text: str) -> int:
    """Parse a duration string into seconds.

    Args:
        text (str): Duration string.

    Returns:
        int: Duration in seconds.
    """
    assert duration_pattern.match(text)
    hours, minutes, seconds = map(int, digits_pattern.findall(text))
    return hours * 3600 + minutes * 60 + seconds


def download(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Download the video for the given sample.

    Args:
        obj (Dict[str, Any]): Sample to download.

    Returns:
        Optional[Dict[str, Any]]: Downloaded sample, or None if download failed.
    """
    url = obj['contentUrl']
    try:
        ret = requests.get(url)
    except:
        return None
    if ret.status_code != 200:
        return None
    obj['duration'] = parse_duration(obj['duration'])
    obj['content'] = ret.content
    return obj


def main(args: Namespace) -> None:
    """Download videos, creating an MDS dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    columns = {
        'videoid': 'str',
        'name': 'str',
        'page_idx': 'str',
        'page_dir': 'str',
        'duration': 'int',
        'contentUrl': 'str',
        'content': 'bytes',
    }
    pool = Pool(args.num_procs)
    todos = each_todo(getattr(args, 'in'))
    if args.limit:
        todos = head(todos, args.limit)
    with MDSWriter(args.out, columns) as out:
        for sample in pool.imap_unordered(download, todos):
            if sample:
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
