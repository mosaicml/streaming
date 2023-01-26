# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Convert and upload LAION-400M parquet shards."""

import json
import os
from argparse import ArgumentParser, Namespace
from time import sleep, time
from typing import Iterator, List, Optional, Union

import numpy as np
from pyarrow import parquet as pq

from streaming import MDSWriter


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--local',
                      type=str,
                      required=True,
                      help='Local directory containing downloaded shards in parquet format.')
    args.add_argument('--remote',
                      type=str,
                      required=True,
                      help='Remote path to upload MDS-formatted shards to.')
    args.add_argument('--keep_parquet',
                      type=int,
                      default=1,
                      help='Whether to keep the parquet shards after conversion (about 10TB).')
    args.add_argument('--keep_mds',
                      type=int,
                      default=1,
                      help='Whether to keep the MDS shards after upload (about 10TB).')
    args.add_argument('--hashes',
                      type=str,
                      default='sha1,xxh64',
                      help='Hashes for validating shards, if any.')
    args.add_argument('--poll_interval',
                      type=float,
                      default=30,
                      help='Interval between polling for newly downloaded shards to process.')
    return args.parse_args()


def each_downloaded_shard(local: str) -> Iterator[int]:
    """Iterate over each downloaded shard.

    Args:
        local (str): Local directory containing shards.

    Returns:
        Iterator[int]: Each downloaded shard ID.
    """
    basenames = set(os.listdir(local))
    count = len(list(filter(lambda s: s.endswith('_stats.json'), basenames)))
    for idx in range(count):
        stats_filename = os.path.join(local, f'{idx:05}_stats.json')
        if not os.path.exists(stats_filename):
            raise RuntimeError('Stats file is missing.')
        yield idx


def get_int(x: Union[float, int]) -> int:
    """Get an int field from pandas.

    Args:
        x (Union[float, int]): The pandas field.

    Returns:
        int: The normalized field.
    """
    if np.isnan(x):
        return 0
    else:
        return int(x)


def get_float(x: float) -> float:
    """Get a float field from pandas.

    Args:
        x (float): The pandas field.

    Returns:
        float: The normalized field.
    """
    return x


def get_bytes(x: Optional[bytes]) -> bytes:
    """Get a bytes field from pandas.

    Args:
        x (bytes, optional): The pandas field.

    Returns:
        float: The normalized field.
    """
    return x or b''


def get_str(x: Optional[str]) -> str:
    """Get a str field from pandas.

    Args:
        x (str, optional): The pandas field.

    Returns:
        str: The normalized field.
    """
    return x or ''


def convert(parquet_filename: str, mds_dirname: str, hashes: List[str]) -> None:
    """Convert a parquet shard to MDS shard with an index.

    Args:
        parquet_filename (str): Filename of the input parquet shard.
        mds_dirname (str): Dirname containing the output MDS shard and index.
        hashes (List[str]): List of hashes to apply to the MDS shard, if any.
    """
    columns = {
        'nsfw': 'str',
        'similarity': 'float64',
        'license': 'str',
        'caption': 'str',
        'url': 'str',
        'key': 'str',
        'status': 'str',
        'error_message': 'str',
        'width': 'int32',
        'height': 'int32',
        'original_width': 'int32',
        'original_height': 'int32',
        'exif': 'str',
        'jpg': 'bytes',
    }
    compression = None  # Don't compress because the vast majority of the data is JPEG.
    size_limit = None  # Put everything in one shard (1:1 mapping of parquet to MDS shards).

    with MDSWriter(mds_dirname, columns, compression, hashes, size_limit) as out:
        table = pq.read_table(parquet_filename)
        nr = table.num_rows
        table = table.to_pandas()
        for i in range(nr):
            x = table.iloc[i]
            sample = {
                'nsfw': get_str(x['NSFW']),
                'similarity': get_float(x['similarity']),
                'license': get_str(x['LICENSE']),
                'caption': get_str(x['caption']),
                'url': get_str(x['url']),
                'key': get_str(x['key']),
                'status': get_str(x['status']),
                'error_message': get_str(x['error_message']),
                'width': get_int(x['width']),
                'height': get_int(x['height']),
                'original_width': get_int(x['original_width']),
                'original_height': get_int(x['original_height']),
                'exif': get_str(x['exif']),
                'jpg': get_bytes(x['jpg']),
            }
            out.write(sample)


def upload(local: str, remote: str) -> None:
    """Upload a shard to remote storage.

    Args:
        local (str): Path on local filesystem.
        remote (str): Path on remote filesystem.
    """
    local = local.replace(' ', '\\ ')
    remote = remote.replace(' ', '\\ ')
    cmd = f'aws s3 cp {local} {remote}'
    if os.system(cmd):
        raise RuntimeError(f'Download failed: {cmd}.')


def convert_and_upload_shards(args: Namespace) -> bool:
    """Process any newly downloaded shards.

    Args:
        args (Namespace): Command-line arguments.

    Returns:
        bool: Whether shard downloading is done.
    """
    hashes = args.hashes.split(',') if args.hashes else []
    for idx in each_downloaded_shard(args.local):
        # If the shard is already done, skip it.
        done_filename = os.path.join(args.local, f'{idx:05}.done')
        if os.path.exists(done_filename):
            print(f'Shard {idx:05}: done')
            continue

        # The shard is not marked done, so possibly convert and definitely upload it.
        parquet_filename = os.path.join(args.local, f'{idx:05}.parquet')
        mds_dirname = os.path.join(args.local, f'{idx:05}.mds')
        mds_shard_filename = os.path.join(mds_dirname, 'shard.00000.mds')
        mds_index_filename = os.path.join(mds_dirname, 'index.json')
        remote_shard_filename = os.path.join(args.remote, f'shard.{idx:05}.mds')
        if os.path.exists(mds_dirname):
            if not os.path.exists(mds_shard_filename):
                raise RuntimeError(f'MDS shard file is missing: {mds_shard_filename}.')
            if not os.path.exists(mds_index_filename):
                raise RuntimeError(f'MDS index file is missing: {mds_index_filename}.')
        else:
            print(f'Shard {idx:05}: converting...')
            convert(parquet_filename, mds_dirname, hashes)
        print(f'Shard {idx:05}: uploading...')
        upload(mds_shard_filename, remote_shard_filename)
        with open(done_filename, 'w') as out:
            out.write('')

        # Clean up after ourselves to save local storage.
        if not args.keep_parquet:
            os.remove(parquet_filename)
        if not args.keep_mds:
            os.remove(mds_shard_filename)
        print(f'Shard {idx:05}: done')

    # Check for the "done" marker.
    filename = os.path.join(args.local, 'done')
    return os.path.exists(filename)


def collect_and_upload_index(args: Namespace) -> None:
    """Finally, collect and upload the index.

    Args:
        args (Namespace): Command-line arguments.
    """
    infos = []
    for idx in each_downloaded_shard(args.local):
        sub_index_filename = os.path.join(args.local, f'{idx:05}.mds', 'index.json')
        obj = json.load(open(sub_index_filename))
        info, = obj['shards']
        infos.append(info)

    obj = {
        'version': 2,
        'shards': infos,
    }
    local = os.path.join(args.local, 'index.json')
    with open(local, 'w') as out:
        json.dump(obj, out)

    remote = os.path.join(args.remote, 'index.json')
    upload(local, remote)


def main(args: Namespace) -> None:
    """Convert and upload shards as they are created.

    Args:
        args (Namespace): Command-line arguments.
    """
    while True:
        last_poll = time()
        is_done = convert_and_upload_shards(args)
        if is_done:
            break
        now = time()
        elapsed = now - last_poll
        if elapsed < args.poll_interval:
            sleep(args.poll_interval - elapsed)

    collect_and_upload_index(args)


if __name__ == '__main__':
    main(parse_args())
