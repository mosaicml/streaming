# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Create dataset index file from input parameters."""
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from core.simulation_dataset import SimulationDataset
from streaming.base import Stream
from typing import Optional
import random
import string

def get_random_foldername():
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))

def create_stream_index(shards: int,
                 samples_per_shard: int,
                 avg_raw_shard_size: int,
                 avg_zip_shard_size: Optional[int]) -> Stream:
    """Create dataset index file from input parameters.

    Args:
        shards (int): Number of shards.
        samples_per_shard (int): Number of samples per shard.
        avg_raw_shard_size (int): Average raw shard size.
        avg_zip_shard_size (int): Average compressed shard size.
    Returns:
        local path to created index file for stream.
    """
    index_data = {
        "version": 2,
    }

    shards_list = []
    for shard_id in range(shards):
        shard_data = {
            "column_encodings": [],
            "column_names": [],
            "column_sizes": [],
            "format": "mds",
            "raw_data": {
                "basename": "shard."+str(shard_id)+".mds",
                "bytes": avg_raw_shard_size,
                "hashes": {}
            },
            "hashes": [],
            "samples": samples_per_shard,
            "size_limit": avg_raw_shard_size,
            "version": 2,
            "zip_data": None,
            "compression": None
        }
        if avg_zip_shard_size is not None:
            shard_data["zip_data"] = {
                "basename": "shard."+str(shard_id)+".mds.zstd",
                "bytes": avg_zip_shard_size,
                "hashes": {}
            }
            shard_data["compression"] = "zstd:16"
        shards_list.append(shard_data)

    index_data["shards"] = shards_list

    # Try making the directory for the stream's index.json file
    foldername = get_random_foldername() + "_indexcreated"
    try:
        os.mkdir(foldername)
    except FileExistsError:
        print("Folder already exists, trying again...")
        foldername = get_random_foldername()
        os.mkdir(foldername)

    with open(foldername+'/index.json', 'w') as f:
        json.dump(index_data, f)
        f.close()
    
    return os.path.join(foldername, 'index.json')