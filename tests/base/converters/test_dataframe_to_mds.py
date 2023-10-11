# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from decimal import Decimal
from typing import Any, Tuple

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

from streaming.base.converters import dataframeToMDS

MY_PREFIX = 'train'
MY_BUCKET = {
    'gs://': 'testing-bucket',
    's3://': 'testing-bucket',
    'oci://': 'testing-bucket',
}
os.environ[
    'OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # set to yes to all fork process in spark calls


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

        if keep_local:
            assert len(os.listdir(out)) > 0, f'{out} is empty'
            for d in os.listdir(out):
                if os.path.isdir(os.path.join(out, d)):
                    assert os.path.exists(os.path.join(
                        out, d, 'index.json')), f'No index.json found in subdirectory {d}'

        if merge_index:
            if keep_local:
                assert os.path.exists(os.path.join(out,
                                                   'index.json')), 'No merged index.json found'
                mgi = json.load(open(os.path.join(out, 'index.json'), 'r'))
                nsamples = 0
                for d in os.listdir(out):
                    sub_dir = os.path.join(out, d)
                    if os.path.isdir(sub_dir):
                        shards = json.load(open(os.path.join(sub_dir, 'index.json'),
                                                'r'))['shards']
                        if shards:
                            nsamples += shards[0]['samples']
                assert nsamples == sum([a['samples'] for a in mgi['shards']])
            if not keep_local:
                assert not os.path.exists(os.path.join(
                    out,
                    'index.json')), 'merged index.json is found even through keep_local = False'
        else:
            assert not os.path.exists(os.path.join(
                out, 'index.json')), 'merged index is created when merge_index=False'

    @pytest.mark.parametrize('use_columns', [True, False])
    def test_end_to_end_conversion_local_decimal(self, decimal_dataframe: Any, use_columns: bool,
                                                 local_remote_dir: Tuple[str, str]):
        out, _ = local_remote_dir
        user_defined_columns = {'id': 'int', 'name': 'str', 'amount': 'str_decimal'}
        mds_kwargs = {'out': out, 'columns': user_defined_columns, 'keep_local': True}

        if use_columns:
            mds_kwargs['columns'] = user_defined_columns

        _, _ = dataframeToMDS(decimal_dataframe, merge_index=True, mds_kwargs=mds_kwargs)
        assert len(os.listdir(out)) > 0, f'{out} is empty'

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

        if keep_local:
            assert len(os.listdir(out)) > 0, f'{out} is empty'
            for d in os.listdir(out):
                if os.path.isdir(os.path.join(out, d)):
                    assert os.path.exists(os.path.join(
                        out, d, 'index.json')), f'No index.json found in subdirectory {d}'

        if merge_index == True:
            if keep_local:
                assert os.path.exists(os.path.join(out,
                                                   'index.json')), 'No merged index.json found'
                mgi = json.load(open(os.path.join(out, 'index.json'), 'r'))
                nsamples = 0
                for d in os.listdir(out):
                    sub_dir = os.path.join(out, d)
                    if os.path.isdir(sub_dir):
                        shards = json.load(open(os.path.join(sub_dir, 'index.json'),
                                                'r'))['shards']
                        if shards:
                            nsamples += shards[0]['samples']
                assert nsamples == sum([a['samples'] for a in mgi['shards']])
            else:
                assert not os.path.exists(os.path.join(
                    out, 'index.json')), 'merged index.json is found even keep_local=False'
        else:
            assert not os.path.exists(os.path.join(
                out, 'index.json')), 'merged index is created when merge_index=False'
