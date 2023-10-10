# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create a dataset index file from input parameters."""

import json
import os
import random
import string
from typing import Optional

from streaming.base.format import get_index_basename


def get_random_foldername():
    """Generate random folder name to store the index file in."""
    return ''.join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        for _ in range(16))


def create_stream_index(shards: int, samples_per_shard: int, avg_raw_shard_size: int,
                        avg_zip_shard_size: Optional[int]) -> str:
    """Create dataset index file from input parameters.

    Args:
        shards (int): Number of shards.
        samples_per_shard (int): Number of samples per shard.
        avg_raw_shard_size (int): Average raw shard size.
        avg_zip_shard_size (int): Average compressed shard size.

    Returns:
        local path to created index file for stream.
    """
    index_data = {'version': 2, 'shards': []}

    shards_list = []
    for _ in range(shards):
        shard_data = {
            'column_encodings': [],
            'column_names': [],
            'column_sizes': [],
            'format': 'mds',
            'raw_data': {
                'basename': '',
                'bytes': avg_raw_shard_size,
                'hashes': {}
            },
            'hashes': [],
            'samples': samples_per_shard,
            'size_limit': avg_raw_shard_size,
            'version': 2,
            'zip_data': None,
            'compression': None
        }
        if avg_zip_shard_size is not None:
            shard_data['zip_data'] = {'basename': '', 'bytes': avg_zip_shard_size, 'hashes': {}}
            shard_data['compression'] = ''
        shards_list.append(shard_data)

    index_data['shards'] = shards_list

    # Try making the directory for the stream's index.json file
    foldername = get_random_foldername() + '_indexcreated'
    try:
        os.mkdir(foldername)
    except FileExistsError:
        print('Folder already exists, trying again...')
        foldername = get_random_foldername()
        os.mkdir(foldername)

    index_basename = get_index_basename()

    with open(f'{foldername}/{index_basename}', 'w') as f:
        json.dump(index_data, f)

    return os.path.join(foldername, index_basename)
