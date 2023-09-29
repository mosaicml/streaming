# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import tempfile
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from typing import Any, List, Optional, Tuple, Union

import pytest

from streaming.base.constant import RESUME
from streaming.base.shared.prefix import _get_path
from streaming.base.storage.download import download_file, list_objects
from streaming.base.storage.upload import CloudUploader
from streaming.base.util import (bytes_to_int, clean_stale_shared_memory, do_merge_index, merge_index,
                                 get_list_arg, number_abbrev_to_int, retry)

MY_PREFIX = 'train'
MY_BUCKET = 'mosaicml-composer-tests'
MANUAL_INTEGRATION_TEST = True
os.environ[
    'OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # set to yes to all fork process in spark calls


@pytest.fixture(scope='function', autouse=True)
def manual_integration_dir() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""
    if MANUAL_INTEGRATION_TEST:
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/gooogle_api_credential.json'
        os.environ[
            'GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/xiaohan.zhang/.mosaic/mosaicml-research-nonprod-027345ddbdfd.json'

    tmp_dir = tempfile.mkdtemp()

    def _method(cloud_prefix: str = 'gs://') -> Tuple[str, str]:
        mock_local_dir = tmp_dir
        mock_remote_dir = os.path.join(cloud_prefix, MY_BUCKET, MY_PREFIX)
        return mock_local_dir, mock_remote_dir

    try:
        yield _method
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # pyright: ignore
        if MANUAL_INTEGRATION_TEST:
            try:
                from google.cloud.storage import Client
                storage_client = Client()
                bucket = storage_client.get_bucket(MY_BUCKET)
                blobs = bucket.list_blobs(prefix=MY_PREFIX)
                for blob in blobs:
                    blob.delete()
            except ImportError:
                raise ImportError('google.cloud.storage is not imported correctly.')


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

def integrity_check(out: Union[str, Tuple[str, str]], keep_local):
    """ Check if merged_index file has integrity
        If merged_index is a cloud url, first download it to a temp local file.
    """

    cu = CloudUploader.get(out, keep_local=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:

        if cu.remote:
            download_file(os.path.join(cu.remote, 'index.json'),
                          os.path.join(temp_dir, 'index.json'),
                          timeout=60)
            local_merged_index_path = os.path.join(temp_dir, 'index.json')
        else:
            local_merged_index_path = os.path.join(cu.local, 'index.json')

        if not keep_local:
            assert not os.path.exists(os.path.join(cu.local, 'index.json'))
            return

        assert os.path.exists(local_merged_index_path)
        merged_index = json.load(open(local_merged_index_path, 'r'))
        n_shard_files = len({b['raw_data']['basename'] for b in merged_index['shards']})
        assert (n_shard_files == 2), 'expected 2 shard files but got {n_shard_files}'

def test_merge_index(manual_integration_dir: Any):
    from decimal import Decimal
    from streaming.base.converters import dataframeToMDS
    from pyspark.sql import SparkSession
    from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

    spark = SparkSession.builder.getOrCreate()  # pyright: ignore
    schema = StructType([
        StructField('id', IntegerType(), nullable=False),
        StructField('name', StringType(), nullable=False),
        StructField('amount', DecimalType(10, 2), nullable=False)
    ])

    data = [(1, 'Alice', Decimal('123.45')), (2, 'Bob', Decimal('67.89')),
            (3, 'Charlie', Decimal('987.65'))]

    df = spark.createDataFrame(data=data, schema=schema).repartition(3)

    _, remote = manual_integration_dir()
    mds_kwargs = {
        'out': remote,
        'columns': {
            'id': 'int',
            'name': 'str'
        },
    }
    print('I am here 0: remote = ', remote)

    mds_path, _ = dataframeToMDS(df, merge_index=False, mds_kwargs=mds_kwargs)

    print('mds_path = ', mds_path)
    print(list_objects("gs://mosaicml-composer-tests/train/"))
    merge_index(remote)

    integrity_check(remote, keep_local=True)


@pytest.mark.parametrize('folder_urls_pattern', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('output_format', ['local', 'remote', 'tuple'])
@pytest.mark.usefixtures('manual_integration_dir')
@pytest.mark.parametrize('keep_local', [True, False])
def test_do_merge_index(manual_integration_dir: Any, keep_local: bool, folder_urls_pattern: int,
                        output_format: str):
    """Validate the final merge index json for following patterns of folder_urls:
        1. All urls are str (local). All urls are accessible locally -> no download
        2. All urls are str (local). At least one url is unaccessible locally -> Error
        3. All urls are tuple (local, remote). All urls are accessible locally -> no download
        4. All urls are tuple (local, remote). At least one url is not accessible locally -> download all
        5. All urls are str (remote) -> download all
    """

    if output_format != 'local':
        if not MANUAL_INTEGRATION_TEST:
            pytest.skip('Require cloud credentials. ' +
                        'skipping. Set MANUAL_INTEGRATION_TEST=True to run the check manually!')
        if output_format == 'remote':
            out = manual_integration_dir()[1]
        else:
            out = manual_integration_dir()
    else:
        out = manual_integration_dir()[0]

    naive_mds_partitions = [
        'tests/resources/naive_MDSdataset/25/', 'tests/resources/naive_MDSdataset/26/',
        'tests/resources/naive_MDSdataset/27/'
    ]

    if folder_urls_pattern == 1:
        folder_urls = [os.getcwd() + '/' + s for s in naive_mds_partitions]
        do_merge_index(folder_urls, out, keep_local=keep_local)

    if folder_urls_pattern == 2:
        with tempfile.TemporaryDirectory() as a_temporary_folder:
            folder_urls = [a_temporary_folder + '/' + s for s in naive_mds_partitions]
            with pytest.raises(FileNotFoundError, match=f'.* does not exist or not accessible.*'):
                do_merge_index(folder_urls, out, keep_local=keep_local)
            return

    if folder_urls_pattern == 3:
        folder_urls = []
        for s in naive_mds_partitions:
            folder_urls.append((os.getcwd() + '/' + s, 'gs://mybucket/' + s))
        do_merge_index(folder_urls, out, keep_local=keep_local)

    if folder_urls_pattern == 4:
        if not MANUAL_INTEGRATION_TEST:
            pytest.skip('Require cloud credentials. ' +
                        'skipping. Set MANUAL_INTEGRATION_TEST=True to run the check manually!')

        with tempfile.TemporaryDirectory() as a_temporary_folder:
            folder_urls = []
            for s in naive_mds_partitions:
                cu_path = (os.getcwd() + '/' + s, 'gs://' + MY_BUCKET + '/' + s)
                cu = CloudUploader.get(cu_path, keep_local=True, exist_ok=True)
                index_json = os.path.join(cu.local, 'index.json')
                if os.path.exists(index_json):
                    cu.upload_file('index.json')
                folder_urls.append((a_temporary_folder, 'gs://' + MY_BUCKET + '/' + s))
            do_merge_index(folder_urls, out, keep_local=keep_local)

    if folder_urls_pattern == 5:
        if not MANUAL_INTEGRATION_TEST:
            pytest.skip('Require cloud credentials. ' +
                        'skipping. Set MANUAL_INTEGRATION_TEST=True to run the check manually!')

        with tempfile.TemporaryDirectory() as a_temporary_folder:
            folder_urls = []
            for s in naive_mds_partitions:
                cu_path = (os.getcwd() + '/' + s, 'gs://' + MY_BUCKET + '/' + s)
                cu = CloudUploader.get(cu_path, keep_local=True, exist_ok=True)
                index_json = os.path.join(cu.local, 'index.json')
                if os.path.exists(index_json):
                    cu.upload_file('index.json')
                folder_urls.append('gs://' + MY_BUCKET + '/' + s)
            do_merge_index(folder_urls, out, keep_local=keep_local)

    integrity_check(out, keep_local=keep_local)


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
