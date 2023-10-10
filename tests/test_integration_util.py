# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import time
import urllib.parse
from typing import Any, Tuple

import pytest

from streaming.base.storage.upload import CloudUploader
from streaming.base.util import merge_index
from tests.test_util import integrity_check

MY_PREFIX = 'train_' + str(time.time())
MY_BUCKET = {
    'gs://': 'mosaicml-composer-tests',
    's3://': 'mosaicml-internal-temporary-composer-testing',
    'oci://': 'mosaicml-internal-checkpoints',
    'dbfs:/Volumes': 'main/mosaic_hackathon/managed-volume',
}
MANUAL_INTEGRATION_TEST = True
os.environ[
    'OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # set to yes to all fork process in spark calls


@pytest.fixture(scope='function', autouse=True)
def manual_integration_dir() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""
    if MANUAL_INTEGRATION_TEST:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(
            os.environ['HOME'], '.mosaic/mosaicml-research-gcs.json')
        os.environ.pop('AWS_ACCESS_KEY_ID', None)
        os.environ.pop('AWS_SECRET_ACCESS_KEY', None)
        os.environ.pop('AWS_SECURITY_TOKEN', None)
        os.environ.pop('AWS_SESSION_TOKEN', None)
        os.environ['AWS_PROFILE'] = 'temporary'

    tmp_dir = tempfile.mkdtemp()

    def _method(cloud_prefix: str = 'gs://') -> Tuple[str, str]:
        mock_local_dir = tmp_dir
        mock_remote_dir = os.path.join(cloud_prefix, MY_BUCKET[cloud_prefix], MY_PREFIX)
        return mock_local_dir, mock_remote_dir

    try:
        yield _method
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # pyright: ignore
        if MANUAL_INTEGRATION_TEST:
            try:
                from google.cloud.storage import Client
                storage_client = Client()
                bucket = storage_client.get_bucket(MY_BUCKET['gs://'])
                blobs = bucket.list_blobs(prefix=MY_PREFIX)
                for blob in blobs:
                    blob.delete()
            except:
                print('tear down gcs test folder failed, continue...')

            try:
                import boto3
                s3 = boto3.client('s3')
                response = s3.list_objects_v2(Bucket=MY_BUCKET['s3://'], Prefix=MY_PREFIX)
                objects_to_delete = [{'Key': obj['Key']} for obj in response.get('Contents', [])]
                if objects_to_delete:
                    s3.delete_objects(Bucket=MY_BUCKET['s3://'],
                                      Delete={'Objects': objects_to_delete})
            except:
                print('tear down s3 test folder failed, continue....')

            try:
                import oci
                client = oci.object_storage.ObjectStorageClient(oci.config.from_file())
                response = client.list_objects(
                    namespace_name=client.get_namespace().data,
                    bucket_name=MY_BUCKET['oci://'],
                    fields=['name'],
                    prefix=MY_PREFIX,
                )

                # Delete the objects
                for obj in response.data.objects:
                    client.delete_object(
                        namespace_name=client.get_namespace().data,
                        bucket_name=MY_BUCKET['oci://'],
                        object_name=obj.name,
                    )
                print(f'Deleted {len(response.data.objects)} objects with prefix: {MY_PREFIX}')

            except:
                print('tear down oci test folder failed, continue...')


@pytest.mark.parametrize('scheme', ['oci://', 'gs://', 's3://', 'dbfs:/Volumes'])
@pytest.mark.parametrize('index_file_urls_pattern', [4, 5])
@pytest.mark.parametrize('out_format', ['remote', 'local', 'tuple'])
@pytest.mark.usefixtures('manual_integration_dir')
@pytest.mark.parametrize('keep_local', [True, False])
@pytest.mark.remote
def test_merge_index_from_list_remote(manual_integration_dir: Any, keep_local: bool,
                                      index_file_urls_pattern: int, out_format: str, scheme: str):
    """Validate the final merge index json for following patterns of index_file_urls:
        1. All urls are str (local). All urls are accessible locally -> no download
        2. All urls are str (local). At least one url is unaccessible locally -> Error
        3. All urls are tuple (local, remote). All urls are accessible locally -> no download
        4. All urls are tuple (local, remote). At least one url is not accessible locally -> download all
        5. All urls are str (remote) -> download all
    """
    from decimal import Decimal

    from pyspark.sql import SparkSession
    from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

    from streaming.base.converters import dataframeToMDS

    def not_merged_index(index_file_path: str, out: str):
        """Check if index_file_path is the merged index at folder out."""
        prefix = str(urllib.parse.urlparse(out).path)
        return os.path.dirname(index_file_path).strip('/') != prefix.strip('/')

    local, remote = manual_integration_dir(scheme)

    if out_format == 'remote':
        out = remote
    else:
        out = (local, remote)
    mds_out = (local, remote)

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

    if index_file_urls_pattern == 4:

        remote_cu = CloudUploader.get(remote, exist_ok=True, keep_local=True)
        remote_index_files = [
            os.path.join(scheme, MY_BUCKET[scheme], o)
            for o in remote_cu.list_objects()
            if o.endswith('.json') and not_merged_index(o, remote)
        ]
        with tempfile.TemporaryDirectory() as a_temporary_folder:
            non_exist_local_files = [
                os.path.join(a_temporary_folder, os.path.basename(s)) for s in local_index_files
            ]
            index_file_urls = list(zip(non_exist_local_files, remote_index_files))
            merge_index(index_file_urls, out, keep_local=keep_local)

    if index_file_urls_pattern == 5:

        remote_cu = CloudUploader.get(remote, exist_ok=True, keep_local=True)
        remote_index_files = [
            os.path.join(scheme, MY_BUCKET[scheme], o)
            for o in remote_cu.list_objects()
            if o.endswith('.json') and not_merged_index(o, remote)
        ]
        merge_index(remote_index_files, out, keep_local=keep_local)

    integrity_check(out, keep_local=keep_local)


@pytest.mark.parametrize('index_file_urls_pattern', [1, 2, 3])
@pytest.mark.usefixtures('manual_integration_dir')
@pytest.mark.parametrize('keep_local', [True, False])
def test_merge_index_from_list_local(manual_integration_dir: Any, keep_local: bool,
                                     index_file_urls_pattern: int):
    """Validate the final merge index json for following patterns of index_file_urls:
        1. All urls are str (local). All urls are accessible locally -> no download
        2. All urls are str (local). At least one url is unaccessible locally -> Error
        3. All urls are tuple (local, remote). All urls are accessible locally -> no download
        4. All urls are tuple (local, remote). At least one url is not accessible locally -> download all
        5. All urls are str (remote) -> download all
    """
    from decimal import Decimal

    from pyspark.sql import SparkSession
    from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

    from streaming.base.converters import dataframeToMDS

    def not_merged_index(index_file_path: str, out: str):
        """Check if index_file_path is the merged index at folder out."""
        prefix = str(urllib.parse.urlparse(out).path)
        return os.path.dirname(index_file_path).strip('/') != prefix.strip('/')

    local, _ = manual_integration_dir()

    mds_out = out = local
    scheme = 's3://'

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
def test_merge_index_from_root_local(manual_integration_dir: Any, n_partitions: int,
                                     keep_local: bool):
    from decimal import Decimal

    from pyspark.sql import SparkSession
    from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

    from streaming.base.converters import dataframeToMDS

    out, _ = manual_integration_dir()

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


@pytest.mark.parametrize('scheme', ['oci://', 'gs://', 's3://', 'dbfs:/Volumes'])
@pytest.mark.parametrize('out_format', ['remote', 'tuple'])
@pytest.mark.parametrize('n_partitions', [1, 2, 3, 4])
@pytest.mark.parametrize('keep_local', [False, True])
@pytest.mark.remote
def test_merge_index_from_root_remote(manual_integration_dir: Any, out_format: str,
                                      n_partitions: int, keep_local: bool, scheme: str):
    from decimal import Decimal

    from pyspark.sql import SparkSession
    from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

    from streaming.base.converters import dataframeToMDS

    if out_format == 'remote':
        _, out = manual_integration_dir(scheme)
    else:
        out = manual_integration_dir(scheme)

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


@pytest.mark.parametrize('scheme', ['dbfs:/Volumes'])
@pytest.mark.parametrize('out_format', ['remote'])  # , 'tuple'])
@pytest.mark.parametrize('n_partitions', [3])  # , 2, 3, 4])
@pytest.mark.parametrize('keep_local', [False])  # , True])
def test_uc_volume(manual_integration_dir: Any, out_format: str, n_partitions: int,
                   keep_local: bool, scheme: str):
    from decimal import Decimal

    from pyspark.sql import SparkSession
    from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

    from streaming.base.converters import dataframeToMDS

    if out_format == 'remote':
        _, out = manual_integration_dir(scheme)
    else:
        out = manual_integration_dir(scheme)

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

    mds_path, _ = dataframeToMDS(df, merge_index=True, mds_kwargs=mds_kwargs)

    with pytest.raises(NotImplementedError,
                       match=f'DatabricksUnityCatalogUploader.list_objects is not implemented.*'):
        merge_index(mds_path, keep_local=keep_local)
