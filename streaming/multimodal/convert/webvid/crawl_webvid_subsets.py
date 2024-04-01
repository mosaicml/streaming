# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Download videos, saving subsets as MDS datasets, then iterate over them together.

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

3. Run this script with flags --csv (CSV) --mds_root (MDS root)
"""

import csv
import os
import re
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Optional

import requests

from streaming import MDSWriter, Stream, StreamingDataset

# For parsing the duration field.
duration_pattern = re.compile('^PT\\d{2}H\\d{2}M\\d{2}S$')
digits_pattern = re.compile('\\d{2}')


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Dataset CSV file from https://m-bain.github.io/webvid-dataset/',
    )
    args.add_argument(
        '--filters',
        type=str,
        default='plane,train,automobile,cat,dog',
        help='Comma-separated list of keywords to filter into sub-datasets',
    )
    args.add_argument(
        '--mds_root',
        type=str,
        required=True,
        help='Root directory path to store the output datasets',
    )
    args.add_argument(
        '--num_procs',
        type=int,
        default=64,
        help='Number of processes to use for downloading videos',
    )
    args.add_argument(
        '--limit',
        type=int,
        default=-1,
        help='Only process the first "limit" number of samples, or all of them if set to -1',
    )
    return args.parse_args()


def get_matches(filters: List[str], text: str) -> List[str]:
    """Get whether one or more filters match the given text.

    Args:
        filters (List[str]): List of substrings to match.
        text (str): Text to search for matches.

    Returns:
        List[str]: List of filters that matched.
    """
    matches = []
    for substr in filters:
        if substr in text:
            matches.append(substr)
    return matches


def each_todo(filename: str, filters: List[str]) -> Iterator[Dict[str, Any]]:
    """Get each sample to download.

    Args:
        filename (str): Path to CSV file containing samples to download.

    Returns:
        Iterator[Dict[str, Any]]: Each sample to download.
    """
    it = csv.reader(open(filename))
    keys = next(it)
    for values in it:
        obj = dict(zip(keys, values))
        matches = get_matches(filters, obj['name'])
        if matches:
            obj['matches'] = matches  # pyright: ignore
            yield obj


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
    # Get the list of strings for filtering into sub-datasets.
    filters = args.filters.split(',') if args.filters else []

    # Define the dataset schema.
    columns = {
        'videoid': 'str',
        'name': 'str',
        'page_dir': 'str',
        'duration': 'int',
        'contentUrl': 'str',
        'content': 'bytes',
    }

    # Get each sample to crawl.
    todos = each_todo(args.csv, filters)
    if args.limit:
        todos = head(todos, args.limit)

    # Crawl samples in parallel.
    pool = Pool(args.num_procs)
    writers = {}
    for sample in pool.imap_unordered(download, todos):
        if not sample:
            continue
        for match in sample['matches']:
            writer = writers.get(match)
            if writer is None:
                dirname = os.path.join(args.mds_root, match)
                writers[match] = writer = MDSWriter(out=dirname, columns=columns)
            writer.write(sample)
    for name in sorted(filters):
        writers[name].finish()

    # Did we crawl enough?
    subsets_present = []
    for substr in filters:
        dirname = os.path.join(args.mds_root, substr)
        subsets_present.append(dirname)

    # Create a Stream per sub-dataset, then pass them to a StreamingDataset.
    streams = []
    for name in sorted(subsets_present):
        dirname = os.path.join(args.mds_root, name)
        stream = Stream(local=dirname, proportion=1 / len(subsets_present))
        streams.append(stream)
    dataset = StreamingDataset(streams=streams, epoch_size=50, batch_size=1)

    # Print the size of each sub-dataset.
    for name, num_samples in zip(sorted(subsets_present), dataset.samples_per_stream):
        print(f'Subset "{name}": {num_samples} samples')

    # Iterate the combined dataset 3 times.
    for epoch in range(3):
        print(f'Epoch {epoch}:')
        for idx, sample in enumerate(dataset):
            text = ' '.join(sample['name'].split())
            print(f'{idx:6} | {text}')


if __name__ == '__main__':
    main(parse_args())
