# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from tempfile import mkdtemp
from typing import Any, Tuple

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from streaming import MDSWriter
from streaming.base.converters import dataframeToMDS

MY_PREFIX = 'train'
MY_BUCKET = 'mosaicml-composer-tests'
LOCAL_MANUAL_TEST = False
os.environ[
    'OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # set to yes to all fork process in spark calls


@pytest.fixture(scope='class', autouse=True)
def remote_local_dir() -> Any:
    """Creates a temporary directory and then deletes it when the calling function is done."""

    os.environ[
        'GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/xiaohan.zhang/.mosaic/mosaicml-research-nonprod-027345ddbdfd.json'

    def _method(cloud_prefix: str = 'gs://') -> Tuple[str, str]:
        mock_local_dir = mkdtemp()
        mock_remote_dir = os.path.join(cloud_prefix, MY_BUCKET, MY_PREFIX)
        return mock_local_dir, mock_remote_dir

    return _method


class TestDataFrameToMDS:

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

        df = spark.createDataFrame(data=data, schema=schema).repartition(2)
        yield df

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    def test_end_to_end_conversion_local_nocolumns(self, dataframe: Any, keep_local: bool,
                                                   merge_index: bool):
        out = mkdtemp()
        mds_kwargs = {
            'out': out,
            'keep_local': keep_local,
            'compression': 'zstd:7',
            'hashes': ['sha1', 'xxh64'],
            'size_limit': 1 << 26
        }

        with pytest.raises(ValueError):
            _ = dataframeToMDS(dataframe.select(col('id'), col('dept'), col('properties')),
                               merge_index=merge_index,
                               sample_ratio=-1.0,
                               mds_kwargs=mds_kwargs)

        _ = dataframeToMDS(dataframe.select(col('id'), col('dept')),
                           merge_index=merge_index,
                           sample_ratio=-1.0,
                           mds_kwargs=mds_kwargs)

        assert (len(os.listdir(out)) > 0), f'{out} is empty'
        for d in os.listdir(out):
            if os.path.isdir(os.path.join(out, d)):
                assert (os.path.exists(os.path.join(
                    out, d, 'index.json'))), f'No index.json found in subdirectory {d}'

        if merge_index == True:
            assert (os.path.exists(os.path.join(out, 'index.json'))), 'No merged index found'
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

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    def test_end_to_end_conversion_local(self, dataframe: Any, keep_local: bool,
                                         merge_index: bool):
        out = mkdtemp()
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

        _ = dataframeToMDS(dataframe,
                           merge_index=merge_index,
                           sample_ratio=-1.0,
                           mds_kwargs=mds_kwargs)

        assert (len(os.listdir(out)) > 0), f'{out} is empty'
        for d in os.listdir(out):
            if os.path.isdir(os.path.join(out, d)):
                assert (os.path.exists(os.path.join(
                    out, d, 'index.json'))), f'No index.json found in subdirectory {d}'

        if merge_index == True:
            assert (os.path.exists(os.path.join(out, 'index.json'))), 'No merged index found'
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
    @pytest.mark.usefixtures('remote_local_dir')
    def test_patch_conversion_local_and_remote(self, dataframe: Any, scheme: str,
                                               merge_index: bool, keep_local: bool,
                                               remote_local_dir: Any):
        if not LOCAL_MANUAL_TEST:
            pytest.skip(
                'Overlap with integration tests. But better figure out how to run this test suite with Mock.'
            )
        mock_local, mock_remote = remote_local_dir(cloud_prefix='gs://')
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

        mds_path = dataframeToMDS(dataframe,
                                  merge_index=merge_index,
                                  sample_ratio=-1.0,
                                  mds_kwargs=mds_kwargs)

        assert out == mds_path, f'returned mds_path: {mds_path} is not the same as out: {out}'

        if keep_local == False:
            assert (not os.path.exists(mds_path[0])), 'local folder were not removed'
            return

        assert (len(os.listdir(mds_path[0])) > 0), f'{mds_path[0]} is empty'
        for d in os.listdir(mds_path[0]):
            if os.path.isdir(os.path.join(mds_path[0], d)):
                assert (os.path.exists(os.path.join(
                    mds_path[0], d, 'index.json'))), f'No index.json found in subdirectory {d}'

        if merge_index == True:
            assert (os.path.exists(os.path.join(mds_path[0],
                                                'index.json'))), 'No merged index found'
        else:
            assert not (os.path.exists(os.path.join(
                mds_path[0], 'index.json'))), 'merged index is created when merge_index=False'

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    @pytest.mark.usefixtures('remote_local_dir')
    def test_integration_conversion_local_and_remote(self, dataframe: Any, remote_local_dir: Any,
                                                     merge_index: bool, keep_local: bool):
        if not LOCAL_MANUAL_TEST:
            pytest.skip('run local only. CI cluster does not have GCS service acct set up.')
        out = remote_local_dir(cloud_prefix='gs://')
        # bucket_name = 'mosaicml-composer-tests'
        mds_kwargs = {
            'out': out,
            'keep_local': keep_local,
            'compression': 'zstd:7',
            'hashes': ['sha1', 'xxh64'],
            'size_limit': 1 << 26
        }

        mds_path = dataframeToMDS(dataframe,
                                  merge_index=merge_index,
                                  sample_ratio=-1.0,
                                  mds_kwargs=mds_kwargs)

        assert out == mds_path, f'returned mds_path: {mds_path} is not the same as out: {out}'

        if keep_local == False:
            assert (not os.path.exists(mds_path[0])), 'local folder were not removed'
            return

        assert (len(os.listdir(mds_path[0])) > 0), f'{mds_path[0]} is empty'
        for d in os.listdir(mds_path[0]):
            if os.path.isdir(os.path.join(mds_path[0], d)):
                assert (os.path.exists(os.path.join(
                    mds_path[0], d, 'index.json'))), f'No index.json found in subdirectory {d}'

        if merge_index == True:
            assert (os.path.exists(os.path.join(mds_path[0],
                                                'index.json'))), 'No merged index found'
        else:
            assert not (
                os.path.exists(os.path.join(mds_path[0], 'index.json'))
            ), f'merged index is created at {mds_path[0]} when merge_index={merge_index} and keep_local={keep_local}'

    @pytest.mark.usefixtures('remote_local_dir')
    def test_integration_conversion_remote_only(self, dataframe: Any, remote_local_dir: Any):
        if not LOCAL_MANUAL_TEST:
            pytest.skip('run local only. CI cluster does not have GCS service acct set up.')
        _, remote = remote_local_dir(cloud_prefix='gs://')
        mds_kwargs = {
            'out': remote,
            'columns': {
                'id': 'str',
                'dept': 'str'
            },
        }

        mds_path = dataframeToMDS(dataframe,
                                  merge_index=True,
                                  sample_ratio=-1.0,
                                  mds_kwargs=mds_kwargs)

        assert len(mds_path) == 2, 'returned mds is a str but should be a tuple (local, remote)'
        assert not (os.path.exists(os.path.join(
            mds_path[0], 'index.json'))), 'Local merged index was not removed successfully'
        assert (len(os.listdir(mds_path[0])) > 0), f'{mds_path[0]} is not empty'

    def test_simple_remote(self, dataframe: Any):
        if not LOCAL_MANUAL_TEST:
            pytest.skip('run local only. CI cluster does not have GCS service acct set up.')

        out = 'gs://mosaicml-composer-tests/test_df2mds'

        with MDSWriter(out=out, columns={'id': 'str', 'dept': 'str'}) as mds_writer:
            d = dataframe.toPandas().to_dict('records')
            for row in d:
                mds_writer.write(row)
