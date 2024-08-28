# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
import time
from typing import Any, Dict, Optional, Tuple

#import pytest

from streaming.base import StreamingDataset, StreamingDataLoader
from streaming.text import StreamingC4
from streaming.vision import StreamingADE20K, StreamingCIFAR10, StreamingCOCO, StreamingImageNet


def get_dataset(name: str,
                split: str,
                shuffle: bool,
                batch_size: Optional[int],
                other_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[int, StreamingDataset]:
    other_kwargs = {} if other_kwargs is None else other_kwargs
    dataset_map = {
        'refinedweb': {
            'local': f'/tmp/test_refinedweb_05May1029',
            'remote': 'dbfs:/Volumes/main/mosaic_hackathon/managed-volume/mds/refinedweb/',
            'num_samples': 20206,
            'class': StreamingDataset,
            'kwargs': {},
        },
        'dummy_table_dbsql': {
            'local': f'/tmp/test_dummy_table_05May1029',
            'remote': 'SELECT * FROM main.streaming.dummy_cpt_table',
            'num_samples': 5,
            'class': StreamingDataset,
            'kwargs': {
                'warehouse_id': "7e083095329f3ca5",
                'catalog': 'main',
                'schema': 'streaming',
            },
        },
        'random_cpt_table_dbsql': {
            'local': f'/tmp/test_random_cpt_table_05May1029',
            'remote': 'SELECT text FROM main.streaming.random_cpt_table',
            'num_samples': 100000,
            'class': StreamingDataset,
            'kwargs': {
                'warehouse_id': "7e083095329f3ca5",
                'catalog': 'main',
                'schema': 'streaming',
            },
        },
        'prompt_response_table_dbsql': {
            'local': f'/tmp/test_prompt_response_table_05May1029',
            'remote': 'SELECT * FROM main.streaming.prompt_response_table_normal_1000000_20000',
            'num_samples': 1000000,
            'class': StreamingDataset,
            'kwargs': {
                'warehouse_id': "7e083095329f3ca5",
                'catalog': 'main',
                'schema': 'streaming',
            },
        },
        'random_large_table': {
            'local': f'/tmp/test_random_large_table_05May1029',
            'remote': 'SELECT * FROM main.streaming.random_large_table',
            'num_samples': 100000,
            'class': StreamingDataset,
            'kwargs': {
                'cluster_id': "0201-234512-tcp9nfat"
            },
        },
        'large_liquid_test_table_08_07': {
            'local': f'/tmp/test_liquid_test_table_05May1029',
            'remote': 'SELECT ss_sold_date_sk, ss_sold_time_sk, ss_item_sk, ss_customer_sk, ss_cdemo_sk FROM auto_maintenance_bugbash.stella.large_liquid_test_table_08_07',
            'num_samples': 100000,
            'class': StreamingDataset,
            'kwargs': {
                'warehouse_id': "7e083095329f3ca5",
                'catalog': 'auto_maintenance_bugbash',
                'schema': 'stella',
            },
        },
        'reddit_table_sparkconnect': {
            'local': f'/tmp/test_random_reddit_table_05May1029',
            'remote': 'SELECT text, added FROM main.reddit.data',
            'num_samples': 378156152,
            'class': StreamingDataset,
            'kwargs': {
                'cluster_id': "0523-224100-tid6mais"
            },
        },
        'reddit_table_dbsql': {
            'local': f'/tmp/test_random_reddit_table_05May1029',
            'remote': 'SELECT text, added FROM main.reddit.data',
            'num_samples': 378156152,
            'class': StreamingDataset,
            'kwargs': {
                'warehouse_id': "89cf2c9b9f9cb3bc",
                'catalog': 'main',
                'schema': 'reddit',
            },
        },
        'reddit_table_dbsql_cachelimit': {
            'local': f'/tmp/test_random_reddit_table_05May1029',
            'remote': 'SELECT text, added FROM main.reddit.data',
            'num_samples': 378156152,
            'class': StreamingDataset,
            'kwargs': {
                'warehouse_id': "89cf2c9b9f9cb3bc",
                'catalog': 'main',
                'schema': 'reddit',
                'cache_limit': '10gb',
            },
        },
        'wiki_table_dbsql_cachelimit': {
            'local': f'/tmp/test_wiki_table_05May1029',
            'remote': 'SELECT id, text FROM main.streaming.wiki_table',
            'num_samples': 378156152,
            'class': StreamingDataset,
            'kwargs': {
                'warehouse_id': "89cf2c9b9f9cb3bc",
                'catalog': 'main',
                'schema': 'streaming',
                'cache_limit': '100mb',
            },
            'shuffle': True,
        },
        'coco_table_dbsql': {
            'local': f'/tmp/test_coco_table_05May1029',
            'remote': 'SELECT data, captions FROM main.streaming.coco_with_meta_and_captions',
            'num_samples': 26688,
            'class': StreamingDataset,
            'kwargs': {
                'warehouse_id': "89cf2c9b9f9cb3bc",
                'catalog': 'main',
                'schema': 'streaming',
                # 'cache_limit': '100mb',
            },
            'shuffle': False,
        },
        'debug_local': {
            'local': f'/tmp/test_random_reddit_table_05May1029',
            'remote': None,
            'num_samples': 378156152,
            'class': StreamingDataset,
            'kwargs': {}
        },
    }
    #if name not in dataset_map and split not in dataset_map[name]['num_samples'][split]:
    #    raise ValueError('Could not load dataset with name={name} and split={split}')

    d = dataset_map[name]
    expected_samples = d['num_samples']
    local = d['local']
    remote = d['remote']
    shuffle = d.get('shuffle', False) or shuffle
    kwargs = {**d['kwargs'], **other_kwargs}
    dataset = d['class'](local=local,
                         remote=remote,
                         split=split,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         **kwargs)
    return (expected_samples, dataset)


def test_streaming_remote_dataset(name: str, split: str) -> None:
    # Build StreamingDataset
    build_start = time.time()
    expected_samples, dataset = get_dataset(name=name,
                                            split=split,
                                            shuffle=False,
                                            batch_size=16)
    build_end = time.time()
    build_dur = build_end - build_start
    print('Built dataset')

    # Test basic iteration
    rcvd_samples = 0
    iter_start = time.time()
    for _ in dataset:
        rcvd_samples += 1

        if (rcvd_samples % 10000 == 0):
            print(f'samples read: {rcvd_samples}')

    iter_end = time.time()
    iter_dur = iter_end - iter_start
    samples_per_sec = rcvd_samples / iter_dur

    # Print debug info
    print(f'received {rcvd_samples} samples')
    print(f'build_dur={build_dur:.2f}s, iter_dur={iter_dur:.2f}, ' +
          f'samples_per_sec={samples_per_sec:.2f}')

    # Test all samples arrived
    assert rcvd_samples >= expected_samples

def test_streaming_remote_dataloader(name: str, split: str) -> None:
    # Build StreamingDataset
    build_start = time.time()
    batch_size = 1
    expected_samples, dataset = get_dataset(name=name,
                                            split=split,
                                            shuffle=False,
                                            batch_size=batch_size)


    data_loader = StreamingDataLoader(dataset,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      prefetch_factor=None,
                                      #persistent_workers=True,
                                      pin_memory=True,
                                      drop_last=True)
    build_end = time.time()
    build_dur = build_end - build_start
    print('Built dataset')

    # Test basic iteration
    rcvd_samples = 0
    iter_start = time.time()

    for epcoh in range(1):
        for batch_idx, data_dict in enumerate(data_loader):
            rcvd_samples += batch_size

            if (rcvd_samples % (10*batch_size) == 0):
                print(f'samples read: {rcvd_samples}')

    iter_end = time.time()
    iter_dur = iter_end - iter_start
    samples_per_sec = rcvd_samples / iter_dur

    # Print debug info
    print(f'received {rcvd_samples} samples')
    print(f'build_dur={build_dur:.2f}s, iter_dur={iter_dur:.2f}, ' +
          f'samples_per_sec={samples_per_sec:.2f}')

    # Test all samples arrived
    assert rcvd_samples >= expected_samples


if __name__ == "__main__":
    from streaming.base.util import clean_stale_shared_memory
    clean_stale_shared_memory()
    #test_streaming_remote_dataset(name = 'refinedweb', split=None)
    #test_streaming_remote_dataset(name = 'dummy_table_dbsql', split=None)
    test_streaming_remote_dataset(name = 'random_cpt_table_dbsql', split=None)
    #test_streaming_remote_dataset(name = 'random_large_table', split=None)
    #test_streaming_remote_dataset(name = 'reddit_table', split=None)
    #test_streaming_remote_dataset(name = 'reddit_table_dbsql', split=None)
    #test_streaming_remote_dataset(name = 'reddit_table_dbsql_cachelimit', split=None)
    #test_streaming_remote_dataset(name = 'coco_table_dbsql', split=None)
    #test_streaming_remote_dataset(name = 'large_liquid_test_table_08_07', split=None)
    #test_streaming_remote_dataset(name = 'prompt_response_table_dbsql', split=None)
    #test_streaming_remote_dataset(name = 'debug_local', split=None)

    #test_streaming_remote_dataloader(name = 'refinedweb', split=None)
    #test_streaming_remote_dataloader(name = 'random_cpt_table_dbsql', split=None)
    #test_streaming_remote_dataloader(name = 'reddit_table_dbsql', split=None)
    #test_streaming_remote_dataloader(name = 'wiki_table_dbsql_cachelimit', split=None)
    #test_streaming_remote_dataloader(name = 'coco_table_dbsql', split=None)

