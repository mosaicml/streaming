# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import time
import urllib.parse
from typing import Tuple, Union

import pytest

from streaming.storage.download import download_file
from streaming.storage.upload import CloudUploader
from streaming.util import merge_index

MY_PREFIX = 'train_' + str(time.time())
MY_BUCKET = {
    'gs://': 'testing-bucket',
    's3://': 'testing-bucket',
    'oci://': 'testing-bucket',
}
os.environ[
    'OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # set to yes to all fork process in spark calls


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

    from streaming.converters import dataframeToMDS

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
    mds_kwargs = {'out': mds_out, 'columns': {'id': 'int', 'name': 'str'}, 'keep_local': True}
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

    from streaming.converters import dataframeToMDS

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

    mds_kwargs = {'out': out, 'columns': {'id': 'int', 'name': 'str'}, 'keep_local': keep_local}

    mds_path, _ = dataframeToMDS(df, merge_index=False, mds_kwargs=mds_kwargs)
    merge_index(mds_path, keep_local=keep_local)
    integrity_check(mds_path, keep_local=keep_local)
