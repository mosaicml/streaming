# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import json
import tempfile
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from typing import Any, List, Optional, Tuple, Union

import pytest

from streaming.base.constant import RESUME
from streaming.base.shared.prefix import _get_path
from streaming.base.util import (bytes_to_int, clean_stale_shared_memory, get_list_arg,
                                 merge_index, do_merge_index, number_abbrev_to_int, retry)


@pytest.mark.parametrize(('text', 'expected_output'), [('hello,world', ['hello', 'world']),
                                                       ('hello', ['hello']), ('', [])])
def test_get_list_arg(text: str, expected_output: List[Optional[str]]):
    output = get_list_arg(text)
    assert output == expected_output


def test_bytes_to_int():
    input_to_expected = [
        ('1234', 1234),
        ('1b', 1),
        ('50b', 50),
        ('50B', 50),
        ('100kb', 102400),
        (' 100 kb', 102400),
        ('75mb', 78643200),
        ('75MB', 78643200),
        ('75 mb ', 78643200),
        ('1.39gb', 1492501135),
        ('1.39Gb', 1492501135),
        ('2tb', 2199023255552),
        ('3pb', 3377699720527872),
        ('1.11eb', 1279742870113600256),
        ('1.09zb', 1286844866581978415104),
        ('2.0yb', 2417851639229258349412352),
        (1234, 1234),
        (1, 1),
        (0.5 * 1024, 512),
        (100 * 1024, 102400),
        (75 * 1024**2, 78643200),
        (75 * 1024 * 1024, 78643200),
        (35.78, 35),
        (325388903.203984, 325388903),
    ]
    for size_pair in input_to_expected:
        output = bytes_to_int(size_pair[0])
        assert output == size_pair[1]


def test_bytes_to_int_Exception():
    input_data = ['', '12kbb', '27mxb', '79kkb']
    for value in input_data:
        with pytest.raises(ValueError, match=f'Unsupported value/suffix.*'):
            _ = bytes_to_int(value)


def test_number_abbrev_to_int():
    input_to_expected = [
        ('1234', 1234),
        ('1k', 1000),
        ('50k', 50000),
        ('50K', 50000),
        ('100k', 100000),
        (' 100 k', 100000),
        ('75m', 75000000),
        ('75M', 75000000),
        ('75 m ', 75000000),
        ('1.39b', 1390000000),
        ('1.39B', 1390000000),
        ('2t', 2000000000000),
        ('3 T', 3000000000000),
        (1234, 1234),
        (1, 1),
        (0.5 * 1000, 500),
        (100 * 1000, 100000),
        (75 * 1000**2, 75000000),
        (75 * 1000 * 1000, 75000000),
        (35.78, 35),
        (325388903.203984, 325388903),
    ]
    for size_pair in input_to_expected:
        output = number_abbrev_to_int(size_pair[0])
        assert output == size_pair[1]


def test_number_abbrev_to_int_Exception():
    input_data = ['', '12kbb', '27mxb', '79bk', '79bb', '79 b    m', 'p 64', '64p']
    for value in input_data:
        with pytest.raises(ValueError, match=f'Unsupported value/suffix.*'):
            _ = number_abbrev_to_int(value)


def test_clean_stale_shared_memory():
    # Create a leaked shared memory
    name = _get_path(0, RESUME)
    _ = BuiltinSharedMemory(name, True, 64)

    # Clean up the stale shared memory
    clean_stale_shared_memory()

    # If clean up is successful, it should raise FileNotFoundError Exception
    with pytest.raises(FileNotFoundError):
        _ = BuiltinSharedMemory(name, False, 64)


@pytest.mark.parametrize('folder_urls_pattern', [1, 2, 3, 4, 5])
@pytest.mark.usefixtures('local_remote_dir')
@pytest.mark.parametrize('keep_local', [True, False])
def test_do_merge_index(local_remote_dir: Tuple[str, str],
                        keep_local: bool,
                        folder_urls_pattern: int):
    """Validate the final merge index json for following patterns of folder_urls:
        1. All urls are str (local). All urls are accessible locally -> no download
        2. All urls are str (local). At least one url is unaccessible locally -> Error
        3. All urls are tuple (local, remote). All urls are accessible locally -> no download
        4. All urls are tuple (local, remote). At least one url is not accessible locally -> download all
        5. All urls are str (remote) -> download all
    """

    naive_mds_partitions = [
        'tests/resources/naive_MDSdataset/25/', 'tests/resources/naive_MDSdataset/26/',
        'tests/resources/naive_MDSdataset/27/'
    ]

    if folder_urls_pattern in [4,5]:
        # Require cloud file transfers. Will be covered by integration tests.
        return

    with tempfile.TemporaryDirectory() as out:
        if folder_urls_pattern == 1:
            folder_urls = [os.getcwd() + '/' + s for s in naive_mds_partitions]
            do_merge_index(folder_urls, out, keep_local=keep_local)


        if folder_urls_pattern == 2:
            folder_urls = [out + '/' + s for s in naive_mds_partitions]
            with pytest.raises(
                   FileNotFoundError,
                   match=f'.* does not exist or not accessible.*'):
                do_merge_index(folder_urls, out, keep_local=keep_local)
            return

        if folder_urls_pattern == 3:
            folder_urls = []
            for s in naive_mds_partitions:
                folder_urls.append((os.getcwd() + '/' + s, 'gs://mybucket/' + s))
            do_merge_index(folder_urls, out, keep_local=keep_local)

        # Integrity checks

        merged_index_path = os.path.join(out, 'index.json')

        if not keep_local:
            assert not os.path.exists(merged_index_path)
            return

        assert os.path.exists(merged_index_path)
        merged_index = json.load(open(merged_index_path, 'r'))
        n_shard_files = len(set([b['raw_data']['basename'] for b in merged_index['shards']]))
        assert(n_shard_files == 2), "expected 2 shard files but got {n_shard_files}"



@pytest.mark.parametrize('with_args', [True, False])
def test_retry(with_args: bool):
    num_tries = 0
    return_after = 2

    if with_args:
        decorator = retry(RuntimeError, num_attempts=3, initial_backoff=0.01, max_jitter=0.01)
        return_after = 2
    else:
        decorator = retry
        # Need to return immediately to avoid timeouts
        return_after = 0

    @decorator
    def flaky_function():
        nonlocal num_tries
        if num_tries < return_after:
            num_tries += 1
            raise RuntimeError('Called too soon!')
        return "Third time's a charm"

    assert flaky_function() == "Third time's a charm"
