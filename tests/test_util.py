# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import time
import urllib.parse
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from typing import List, Optional, Tuple, Union

import pytest

from streaming.base.constant import RESUME
from streaming.base.shared.prefix import _get_path
from streaming.base.storage.download import download_file
from streaming.base.storage.upload import CloudUploader
from streaming.base.util import (bytes_to_int, clean_stale_shared_memory, get_list_arg,
                                 merge_index, number_abbrev_to_int, retry)

MY_PREFIX = 'train_' + str(time.time())
MY_BUCKET = {
    'gs://': 'testing-bucket',
    's3://': 'testing-bucket',
    'oci://': 'testing-bucket',
}
os.environ[
    'OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # set to yes to all fork process in spark calls


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
    # Clean up the stale shared memory
    clean_stale_shared_memory()

    # Create a leaked shared memory
    name = _get_path(0, RESUME)
    _ = BuiltinSharedMemory(name, True, 64)

    # Clean up the stale shared memory
    clean_stale_shared_memory()

    # If clean up is successful, it should raise FileNotFoundError Exception
    with pytest.raises(FileNotFoundError):
        _ = BuiltinSharedMemory(name, False, 64)


def integrity_check(out: Union[str, Tuple[str, str]],
                    keep_local: bool,
                    expected_n_shard_files: int = -1):
    """Check if merged_index file has integrity
        If merged_index is a cloud url, first download it to a temp local file.

    Args:
        out (Union[str, Tuple[str,str]]): folder that merged index.json resides
        keep_local: whether to check local file
        expected_n_shard_files (int): If -1, find the number in `out` with get_expected()
    """

    def get_expected(mds_root: str):
        n_shard_files = 0
        cu = CloudUploader.get(mds_root, exist_ok=True, keep_local=True)
        for o in cu.list_objects():
            if o.endswith('.mds'):
                n_shard_files += 1
        return n_shard_files

    cu = CloudUploader.get(out, keep_local=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        if cu.remote:
            download_file(os.path.join(cu.remote, 'index.json'),
                          os.path.join(temp_dir, 'index.json'),
                          timeout=60)
            if expected_n_shard_files == -1:
                expected_n_shard_files = get_expected(cu.remote)
            local_merged_index_path = os.path.join(temp_dir, 'index.json')
        else:
            local_merged_index_path = os.path.join(cu.local, 'index.json')
            if expected_n_shard_files == -1:
                expected_n_shard_files = get_expected(cu.local)

        if not keep_local:
            assert not os.path.exists(os.path.join(cu.local, 'index.json'))
            return

        assert os.path.exists(
            local_merged_index_path
        ), f'{local_merged_index_path} does not exist when keep_local is {keep_local}'
        merged_index = json.load(open(local_merged_index_path, 'r'))
        n_shard_files = len({b['raw_data']['basename'] for b in merged_index['shards']})
        assert n_shard_files == expected_n_shard_files, f'expected {expected_n_shard_files} shard files but got {n_shard_files}'


@pytest.mark.parametrize('scheme', ['gs', 's3', 'oci', 'dbfs'])
def test_format_remote_index_files(scheme: str):
    """Validate the format of remote index files."""
    from streaming.base.util import _format_remote_index_files

    if scheme == 'dbfs':
        remote = os.path.join('dbfs:/', 'Volumes')
    else:
        full_scheme = scheme + '://'
        remote = os.path.join(full_scheme, MY_BUCKET[full_scheme], MY_PREFIX)

    index_files = [
        os.path.join('folder_1', 'index.json'),
        os.path.join('folder_2', 'index.json'),
        os.path.join('folder_3', 'index.json'),
    ]
    remote_index_files = _format_remote_index_files(remote, index_files)

    assert len(remote_index_files) == len(index_files)
    for file in remote_index_files:
        obj = urllib.parse.urlparse(file)
        assert obj.scheme == scheme


@pytest.mark.parametrize('index_file_urls_pattern', [1, 2, 3])
@pytest.mark.parametrize('keep_local', [True, False])
@pytest.mark.parametrize('scheme', ['gs://', 's3://', 'oci://'])
def test_merge_index_from_list_local(local_remote_dir: Tuple[str, str], keep_local: bool,
                                     index_file_urls_pattern: int, scheme: str):
    """Validate the final merge index json for following patterns of index_file_urls:
    1. All URLs are str (local). All URLs are accessible locally -> no download
    2. All URLs are str (local). At least one url is unaccessible locally -> Error
    3. All URLs are tuple (local, remote). All URLs are accessible locally -> no download
    4. All URLs are tuple (local, remote). At least one url is not accessible locally -> download all
    5. All URLs are str (remote) -> download all
    """
    from decimal import Decimal

    from pyspark.sql import SparkSession
    from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

    from streaming.base.converters import dataframeToMDS

    def not_merged_index(index_file_path: str, out: str):
        """Check if index_file_path is the merged index at folder out."""
        prefix = str(urllib.parse.urlparse(out).path)
        return os.path.dirname(index_file_path).strip('/') != prefix.strip('/')

    local, _ = local_remote_dir

    mds_out = out = local

    spark = SparkSession.builder.getOrCreate()  # pyright: ignore
    schema = StructType([
        StructField('id', IntegerType(), nullable=False),
        StructField('name', StringType(), nullable=False),
        StructField('amount', DecimalType(10, 2), nullable=False)
    ])
    data = [(1, 'Alice', Decimal('123.45')), (2, 'Bob', Decimal('67.89')),
            (3, 'Charlie', Decimal('987.65'))]
    df = spark.createDataFrame(data=data, schema=schema).repartition(3)
    mds_kwargs = {'out': mds_out, 'columns': {'id': 'int32', 'name': 'str'}, 'keep_local': True}
    dataframeToMDS(df, merge_index=False, mds_kwargs=mds_kwargs)

    local_cu = CloudUploader.get(local, exist_ok=True, keep_local=True)
    local_index_files = [
        o for o in local_cu.list_objects() if o.endswith('.json') and not_merged_index(o, local)
    ]

    if index_file_urls_pattern == 1:
        merge_index(local_index_files, out, keep_local=keep_local)

    if index_file_urls_pattern == 2:
        with tempfile.TemporaryDirectory() as a_temporary_folder:
            index_file_urls = [
                os.path.join(a_temporary_folder, os.path.basename(s)) for s in local_index_files
            ]
            with pytest.raises(RuntimeError, match=f'.*Failed to download index.json.*'):
                merge_index(index_file_urls, out, keep_local=keep_local)

        with tempfile.TemporaryDirectory() as a_temporary_folder:
            index_file_urls = [(os.path.join(a_temporary_folder, os.path.basename(s)), '')
                               for s in local_index_files]
            with pytest.raises(FileNotFoundError, match=f'.*Check data availability!.*'):
                merge_index(index_file_urls, out, keep_local=keep_local)
            return

    if index_file_urls_pattern == 3:
        remote_index_files = [
            os.path.join(scheme, MY_BUCKET[scheme], MY_PREFIX, os.path.basename(o))
            for o in local_index_files
            if o.endswith('.json') and not_merged_index(o, local)
        ]
        index_file_urls = list(zip(local_index_files, remote_index_files))
        merge_index(index_file_urls, out, keep_local=keep_local)

    integrity_check(out, keep_local=keep_local)


@pytest.mark.parametrize('n_partitions', [1, 2, 3, 4])
@pytest.mark.parametrize('keep_local', [False, True])
def test_merge_index_from_root_local(local_remote_dir: Tuple[str, str], n_partitions: int,
                                     keep_local: bool):
    from decimal import Decimal

    from pyspark.sql import SparkSession
    from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

    from streaming.base.converters import dataframeToMDS

    out, _ = local_remote_dir

    spark = SparkSession.builder.getOrCreate()  # pyright: ignore
    schema = StructType([
        StructField('id', IntegerType(), nullable=False),
        StructField('name', StringType(), nullable=False),
        StructField('amount', DecimalType(10, 2), nullable=False)
    ])

    data = [(1, 'Alice', Decimal('123.45')), (2, 'Bob', Decimal('67.89')),
            (3, 'Charlie', Decimal('987.65'))]

    df = spark.createDataFrame(data=data, schema=schema).repartition(n_partitions)

    mds_kwargs = {'out': out, 'columns': {'id': 'int32', 'name': 'str'}, 'keep_local': keep_local}

    mds_path, _ = dataframeToMDS(df, merge_index=False, mds_kwargs=mds_kwargs)
    merge_index(mds_path, keep_local=keep_local)
    integrity_check(mds_path, keep_local=keep_local)


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
