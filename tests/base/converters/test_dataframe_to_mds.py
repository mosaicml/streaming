# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from decimal import Decimal
from tempfile import mkdtemp
from typing import Any, Tuple

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

from streaming import MDSWriter
from streaming.base.converters import dataframeToMDS

MY_PREFIX = 'train'
MY_BUCKET = 'mosaicml-composer-tests'
MANUAL_INTEGRATION_TEST = False
os.environ[
    'OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # set to yes to all fork process in spark calls


@pytest.fixture(scope='function', autouse=True)
def manual_integration_dir() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""
    if MANUAL_INTEGRATION_TEST:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/gooogle_api_credential.json'

    tmp_dir = mkdtemp()

    def _method(cloud_prefix: str = 'gs://') -> Tuple[str, str]:
        mock_local_dir = tmp_dir  # mkdtemp()
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


class TestDataFrameToMDS:

    @pytest.fixture
    def decimal_dataframe(self):
        spark = SparkSession.builder.getOrCreate()  # pyright: ignore

        schema = StructType([
            StructField('id', IntegerType(), nullable=False),
            StructField('name', StringType(), nullable=False),
            StructField('amount', DecimalType(10, 2), nullable=False)
        ])

        data = [(1, 'Alice', Decimal('123.45')), (2, 'Bob', Decimal('67.89')),
                (3, 'Charlie', Decimal('987.65'))]
        df = spark.createDataFrame(data, schema)

        yield df

    @pytest.fixture
    def dataframe(self):
        spark = SparkSession.builder.getOrCreate()  # pyright: ignore

        data = [('36636', 'Finance', (3000, 'USA')), ('40288', 'Finance', (5000, 'IND')),
                ('42114', 'Sales', (3900, 'USA')), ('39192', 'Marketing', (2500, 'CAN')),
                ('34534', 'Sales', (6500, 'USA'))]
        schema = StructType([
            StructField('id', StringType(), True),
            StructField('dept', StringType(), True),
            StructField(
                'properties',
                StructType([
                    StructField('salary', IntegerType(), True),
                    StructField('location', StringType(), True)
                ]))
        ])

        df = spark.createDataFrame(data=data, schema=schema).repartition(3)
        yield df

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    def test_end_to_end_conversion_local_nocolumns(self, dataframe: Any, keep_local: bool,
                                                   merge_index: bool,
                                                   local_remote_dir: Tuple[str, str]):
        out, _ = local_remote_dir
        mds_kwargs = {
            'out': out,
            'keep_local': keep_local,
        }

        with pytest.raises(ValueError, match=f'.*is not supported by MDSWriter.*'):
            _, _ = dataframeToMDS(dataframe.select(col('id'), col('dept'), col('properties')),
                                  merge_index=merge_index,
                                  mds_kwargs=mds_kwargs)

        _, _ = dataframeToMDS(dataframe.select(col('id'), col('dept')),
                              merge_index=merge_index,
                              mds_kwargs=mds_kwargs)

        assert (len(os.listdir(out)) > 0), f'{out} is empty'
        for d in os.listdir(out):
            if os.path.isdir(os.path.join(out, d)):
                assert (os.path.exists(os.path.join(
                    out, d, 'index.json'))), f'No index.json found in subdirectory {d}'

        if merge_index:
            assert (os.path.exists(os.path.join(out, 'index.json'))), 'No merged index.json found'
            mgi = json.load(open(os.path.join(out, 'index.json'), 'r'))
            nsamples = 0
            for d in os.listdir(out):
                sub_dir = os.path.join(out, d)
                if os.path.isdir(sub_dir):
                    shards = json.load(open(os.path.join(sub_dir, 'index.json'), 'r'))['shards']
                    if shards:
                        nsamples += shards[0]['samples']
            assert (nsamples == sum([a['samples'] for a in mgi['shards']]))
        else:
            assert not (os.path.exists(os.path.join(
                out, 'index.json'))), 'merged index is created when merge_index=False'

    @pytest.mark.parametrize('use_columns', [True, False])
    def test_end_to_end_conversion_local_decimal(self, decimal_dataframe: Any, use_columns: bool,
                                                 local_remote_dir: Tuple[str, str]):
        out, _ = local_remote_dir
        user_defined_columns = {'id': 'int', 'name': 'str', 'amount': 'str_decimal'}
        mds_kwargs = {
            'out': out,
            'columns': user_defined_columns,
        }

        if use_columns:
            mds_kwargs['columns'] = user_defined_columns

        _, _ = dataframeToMDS(decimal_dataframe, merge_index=True, mds_kwargs=mds_kwargs)
        assert (len(os.listdir(out)) > 0), f'{out} is empty'

    def test_user_defined_columns(self, dataframe: Any, local_remote_dir: Tuple[str, str]):
        out, _ = local_remote_dir
        user_defined_columns = {'idd': 'str', 'dept': 'str'}
        mds_kwargs = {
            'out': out,
            'columns': user_defined_columns,
        }
        with pytest.raises(ValueError, match=f'.*is not a column of input dataframe.*'):
            _, _ = dataframeToMDS(dataframe, merge_index=False, mds_kwargs=mds_kwargs)

        user_defined_columns = {'id': 'strr', 'dept': 'str'}

        mds_kwargs = {
            'out': out,
            'columns': user_defined_columns,
        }
        with pytest.raises(ValueError, match=f'.* is not supported by MDSWriter.*'):
            _, _ = dataframeToMDS(dataframe, merge_index=False, mds_kwargs=mds_kwargs)

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    def test_end_to_end_conversion_local(self, dataframe: Any, keep_local: bool, merge_index: bool,
                                         local_remote_dir: Tuple[str, str]):
        out, _ = local_remote_dir
        mds_kwargs = {
            'out': out,
            'columns': {
                'id': 'str',
                'dept': 'str'
            },
            'keep_local': keep_local,
            'compression': 'zstd:7',
            'hashes': ['sha1', 'xxh64'],
            'size_limit': 1 << 26
        }

        _, _ = dataframeToMDS(dataframe, merge_index=merge_index, mds_kwargs=mds_kwargs)

        assert (len(os.listdir(out)) > 0), f'{out} is empty'
        for d in os.listdir(out):
            if os.path.isdir(os.path.join(out, d)):
                assert (os.path.exists(os.path.join(
                    out, d, 'index.json'))), f'No index.json found in subdirectory {d}'

        if merge_index == True:
            assert (os.path.exists(os.path.join(out, 'index.json'))), 'No merged index.json found'
            mgi = json.load(open(os.path.join(out, 'index.json'), 'r'))
            nsamples = 0
            for d in os.listdir(out):
                sub_dir = os.path.join(out, d)
                if os.path.isdir(sub_dir):
                    shards = json.load(open(os.path.join(sub_dir, 'index.json'), 'r'))['shards']
                    if shards:
                        nsamples += shards[0]['samples']
            assert (nsamples == sum([a['samples'] for a in mgi['shards']]))
        else:
            assert not (os.path.exists(os.path.join(
                out, 'index.json'))), 'merged index is created when merge_index=False'

    @pytest.mark.parametrize('scheme', ['gs'])
    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    @pytest.mark.usefixtures('manual_integration_dir')
    def test_patch_conversion_local_and_remote(self, dataframe: Any, scheme: str,
                                               merge_index: bool, keep_local: bool,
                                               manual_integration_dir: Any):
        if not MANUAL_INTEGRATION_TEST:
            pytest.skip(
                'Overlap with integration tests. But better figure out how to run this test ' +
                'suite with Mock.')
        mock_local, mock_remote = manual_integration_dir()
        out = (mock_local, mock_remote)
        mds_kwargs = {
            'out': out,
            'columns': {
                'id': 'str',
                'dept': 'str'
            },
            'keep_local': keep_local,
            'compression': 'zstd:7',
            'hashes': ['sha1', 'xxh64'],
            'size_limit': 1 << 26
        }

        mds_path, fail_count = dataframeToMDS(dataframe,
                                              merge_index=merge_index,
                                              mds_kwargs=mds_kwargs)

        assert (fail_count == 0), 'some records were not converted correctly'
        assert out == mds_path, f'returned mds_path: {mds_path} is not the same as out: {out}'

        if not keep_local:
            assert (not os.path.exists(mds_path[0])), 'local folder were not removed'
            return

        assert (len(os.listdir(mds_path[0])) > 0), f'{mds_path[0]} is empty'
        for d in os.listdir(mds_path[0]):
            if os.path.isdir(os.path.join(mds_path[0], d)):
                assert (os.path.exists(os.path.join(
                    mds_path[0], d, 'index.json'))), f'No index.json found in subdirectory {d}'

        if merge_index == True:
            assert (os.path.exists(os.path.join(mds_path[0],
                                                'index.json'))), 'No merged index.json found'
        else:
            assert not (os.path.exists(os.path.join(
                mds_path[0], 'index.json'))), 'merged index is created when merge_index=False'

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    @pytest.mark.usefixtures('manual_integration_dir')
    def test_integration_conversion_local_and_remote(self, dataframe: Any,
                                                     manual_integration_dir: Any,
                                                     merge_index: bool, keep_local: bool):
        if not MANUAL_INTEGRATION_TEST:
            pytest.skip('run local only. CI cluster does not have GCS service acct set up.')
        out = manual_integration_dir()
        mds_kwargs = {
            'out': out,
            'columns': {
                'id': 'str',
                'dept': 'str'
            },
            'keep_local': keep_local,
            'compression': 'zstd:7',
            'hashes': ['sha1', 'xxh64'],
            'size_limit': 1 << 26
        }

        mds_path, _ = dataframeToMDS(dataframe, merge_index=merge_index, mds_kwargs=mds_kwargs)

        assert out == mds_path, f'returned mds_path: {mds_path} is not the same as out: {out}'

        if not keep_local:
            assert (not os.path.exists(mds_path[0])), 'local folder were not removed'
            return

        assert (len(os.listdir(mds_path[0])) > 0), f'{mds_path[0]} is empty'
        for d in os.listdir(mds_path[0]):
            if os.path.isdir(os.path.join(mds_path[0], d)):
                assert (os.path.exists(os.path.join(
                    mds_path[0], d, 'index.json'))), f'No index.json found in subdirectory {d}'

        if merge_index == True:
            assert (os.path.exists(os.path.join(mds_path[0],
                                                'index.json'))), 'No merged index.json found'
        else:
            assert not (os.path.exists(os.path.join(mds_path[0], 'index.json'))), (
                f'merged index is created at {mds_path[0]} when merge_index={merge_index} and ' +
                f'keep_local={keep_local}')

    @pytest.mark.usefixtures('manual_integration_dir')
    def test_integration_conversion_remote_only(self, dataframe: Any, manual_integration_dir: Any):
        if not MANUAL_INTEGRATION_TEST:
            pytest.skip('run local only. CI cluster does not have GCS service acct set up.')
        _, remote = manual_integration_dir()
        mds_kwargs = {
            'out': remote,
            'columns': {
                'id': 'str',
                'dept': 'str'
            },
        }

        mds_path, _ = dataframeToMDS(dataframe, merge_index=True, mds_kwargs=mds_kwargs)

        assert len(mds_path) == 2, 'returned mds is a str but should be a tuple (local, remote)'
        assert not (os.path.exists(os.path.join(
            mds_path[0], 'index.json'))), 'Local merged index was not removed successfully'
        assert (len(os.listdir(mds_path[0])) > 0), f'{mds_path[0]} is not empty'

    def test_simple_remote(self, dataframe: Any):
        if not MANUAL_INTEGRATION_TEST:
            pytest.skip('run local only. CI cluster does not have GCS service acct set up.')

        out = 'gs://mosaicml-composer-tests/test_df2mds'

        with MDSWriter(out=out, columns={'id': 'str', 'dept': 'str'}) as mds_writer:
            d = dataframe.toPandas().to_dict('records')
            for row in d:
                mds_writer.write(row)
