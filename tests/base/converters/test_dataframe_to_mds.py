# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from decimal import Decimal
from typing import Any

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (ArrayType, BinaryType, ByteType, DateType, DecimalType, DoubleType,
                               FloatType, IntegerType, LongType, MapType, ShortType, StringType,
                               StructField, StructType, TimestampType)

from joshua.base.converters import dataframe_to_mds, infer_dataframe_schema, is_json_compatible

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
    def complex_dataframe(self):
        spark = SparkSession.builder.getOrCreate()  # pyright: ignore
        message_schema = ArrayType(
            StructType([
                StructField('role', StringType(), nullable=True),
                StructField('content', StringType(), nullable=True)
            ]))
        data = [[{
            'role': 'system',
            'content': 'Hello, World!'
        }, {
            'role': 'user',
            'content': 'Hi, MPT!'
        }, {
            'role': 'assistant',
            'content': 'Hi, user!'
        }], [{
            'role': 'user',
            'content': 'Hi, MPT!'
        }, {
            'role': 'assistant',
            'content': 'Hi, user!'
        }]]

        df = spark.createDataFrame(data, schema=message_schema)
        df = df.withColumnRenamed('value', 'messages')
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

    @pytest.fixture
    def array_dataframe(self):
        spark = SparkSession.builder.getOrCreate()  # pyright: ignore

        data = [([1, 2, 3], [1.0, 2.0, 3.0], [1e10, 1e11, 1e12]),
                ([4, 5, 6], [4.0, 5.0, 6.0], [2e10, 2e11, 2e12]),
                ([7, 8, 9], [7.0, 8.0, 9.0], [3e10, 3e11, 3e12])]

        schema = StructType([
            StructField('id', ArrayType(ShortType()), True),
            StructField('dept', ArrayType(FloatType()), True),
            StructField('properties', ArrayType(DoubleType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema).repartition(3)
        yield df

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    def test_end_to_end_conversion_local_nocolumns(self, dataframe: Any, keep_local: bool,
                                                   merge_index: bool,
                                                   local_remote_dir: tuple[str, str]):
        out, _ = local_remote_dir
        mds_kwargs = {
            'out': out,
            'keep_local': keep_local,
        }

        # Automatic schema inference for ``properties`` is not available
        with pytest.raises(ValueError, match=f'.*is not supported by dataframe_to_mds.*'):
            _ = dataframe_to_mds(dataframe.select(col('id'), col('dept'), col('properties')),
                                 merge_index=merge_index,
                                 mds_kwargs=mds_kwargs)

        _ = dataframe_to_mds(dataframe.select(col('id'), col('dept')),
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
                assert os.path.exists(
                    os.path.join(out, 'index.json')
                ), 'merged index.json was not found keep_local is False but no remote exists'
        else:
            assert not os.path.exists(os.path.join(
                out, 'index.json')), 'merged index is created when merge_index=False'

    @pytest.mark.parametrize('use_columns', [True, False])
    def test_end_to_end_conversion_local_decimal(self, decimal_dataframe: Any, use_columns: bool,
                                                 local_remote_dir: tuple[str, str]):
        out, _ = local_remote_dir
        user_defined_columns = {'id': 'int32', 'name': 'str', 'amount': 'str_decimal'}
        mds_kwargs = {'out': out, 'columns': user_defined_columns, 'keep_local': True}

        if use_columns:
            mds_kwargs['columns'] = user_defined_columns

        _ = dataframe_to_mds(decimal_dataframe, merge_index=True, mds_kwargs=mds_kwargs)
        assert len(os.listdir(out)) > 0, f'{out} is empty'

    def test_user_defined_columns(self, dataframe: Any, local_remote_dir: tuple[str, str]):
        out, _ = local_remote_dir
        user_defined_columns = {'idd': 'str', 'dept': 'str'}
        mds_kwargs = {
            'out': out,
            'columns': user_defined_columns,
        }
        with pytest.raises(ValueError, match=f'.*is not a column of input dataframe.*'):
            _ = dataframe_to_mds(dataframe, merge_index=False, mds_kwargs=mds_kwargs)

        user_defined_columns = {'id': 'strr', 'dept': 'str'}

        mds_kwargs = {
            'out': out,
            'columns': user_defined_columns,
        }
        # 'strr' is not a valid mds dtype
        with pytest.raises(ValueError, match=f'.* is not supported by dataframe_to_mds.*'):
            _ = dataframe_to_mds(dataframe, merge_index=False, mds_kwargs=mds_kwargs)

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    def test_end_to_end_conversion_local(self, dataframe: Any, keep_local: bool, merge_index: bool,
                                         local_remote_dir: tuple[str, str]):
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

        _ = dataframe_to_mds(dataframe, merge_index=merge_index, mds_kwargs=mds_kwargs)

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
                assert os.path.exists(
                    os.path.join(out, 'index.json')
                ), 'merged index.json was not found when keep_local is False but no remote part exists'
        else:
            assert not os.path.exists(os.path.join(
                out, 'index.json')), 'merged index is created when merge_index=False'

    def test_successful_type_mapping(self):
        spark = SparkSession.builder.getOrCreate()  # pyright: ignore
        test_schema = StructType([
            StructField('byte_col', ByteType(), True),
            StructField('short_col', ShortType(), True),
            StructField('int_col', IntegerType(), True),
            StructField('long_col', LongType(), True),
            StructField('float_col', FloatType(), True),
            StructField('double_col', DoubleType(), True),
            StructField('decimal_col', DecimalType(10, 2), True),
            StructField('string_col', StringType(), True),
            StructField('binary_col', BinaryType(), True),
            StructField('array_int_col', ArrayType(IntegerType()), True),
        ])
        test_data = [(1, 2, 3, 4, 5.0, 6.0, Decimal('123.45'), 'eight', bytearray(b'nine'),
                      [1, 2, 3])]
        test_df = spark.createDataFrame(test_data, schema=test_schema)
        expected_mappings = {
            'byte_col': 'uint8',
            'short_col': 'uint16',
            'int_col': 'int32',
            'long_col': 'int64',
            'float_col': 'float32',
            'double_col': 'float64',
            'decimal_col': 'str_decimal',
            'string_col': 'str',
            'binary_col': 'bytes',
            'array_int_col': 'ndarray:int32',
        }
        assert infer_dataframe_schema(test_df) == expected_mappings

    def test_unsupported_type_raises_value_error(self):
        spark = SparkSession.builder.getOrCreate()  # pyright: ignore

        unsupported_schema = StructType([StructField('unsupported_col', TimestampType(), True)])
        df_with_unsupported_type = spark.createDataFrame([], schema=unsupported_schema)

        with pytest.raises(ValueError):
            infer_dataframe_schema(df_with_unsupported_type)

    @pytest.mark.parametrize('keep_local', [True, False])
    @pytest.mark.parametrize('merge_index', [True, False])
    def test_array_end_to_end_conversion_local_nocolumns(self, array_dataframe: Any,
                                                         keep_local: bool, merge_index: bool,
                                                         local_remote_dir: tuple[str, str]):
        out, _ = local_remote_dir
        mds_kwargs = {
            'out': out,
            'keep_local': keep_local,
        }

        _ = dataframe_to_mds(array_dataframe.select(col('id'), col('dept'), col('properties')),
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
                assert os.path.exists(
                    os.path.join(out, 'index.json')
                ), 'merged index.json was not found keep_local is False but no remote exists'
        else:
            assert not os.path.exists(os.path.join(
                out, 'index.json')), 'merged index is created when merge_index=False'

    def test_array_udf_correct_columns(self,
                                       array_dataframe: Any,
                                       local_remote_dir: tuple[str, str],
                                       keep_local: bool = True,
                                       merge_index: bool = True):
        out, _ = local_remote_dir
        mds_kwargs = {
            'out': out,
            'keep_local': keep_local,
            'columns': {
                'id': 'ndarray:int16:3',
                'dept': 'ndarray:float32:3',
                'properties': 'ndarray:float64:3'
            },
        }

        _ = dataframe_to_mds(array_dataframe.select(col('id'), col('dept'), col('properties')),
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
                assert os.path.exists(
                    os.path.join(out, 'index.json')
                ), 'merged index.json was not found keep_local is False but no remote exists'
        else:
            assert not os.path.exists(os.path.join(
                out, 'index.json')), 'merged index is created when merge_index=False'

    def test_array_udf_wrong_columns(self,
                                     array_dataframe: Any,
                                     local_remote_dir: tuple[str, str],
                                     keep_local: bool = True,
                                     merge_index: bool = True):
        out, _ = local_remote_dir
        mds_kwargs = {
            'out': out,
            'keep_local': keep_local,
            'columns': {
                'id': 'ndarray:int32:3',
                'dept': 'ndarray:float64:3',
                'properties': 'ndarray:float64:3'
            },
        }

        with pytest.raises(ValueError, match=f'.*Mismatched types:.*'):
            _ = dataframe_to_mds(array_dataframe.select(col('id'), col('dept'), col('properties')),
                                 merge_index=merge_index,
                                 mds_kwargs=mds_kwargs)

    def test_is_json_compatible(self):

        message_schema = ArrayType(
            StructType([
                StructField('role', StringType(), nullable=True),
                StructField('content', StringType(), nullable=True)
            ]))

        prompt_response_schema = StructType([
            StructField('prompt', StringType(), nullable=True),
            StructField('response', StringType(), nullable=True)
        ])

        combined_schema = StructType([
            StructField(
                'prompt_response',
                StructType([
                    StructField('prompt', StringType(), True),
                    StructField('response', StringType(), True)
                ]), True),
            StructField(
                'messages',
                ArrayType(
                    StructType([
                        StructField('role', StringType(), True),
                        StructField('content', StringType(), True)
                    ]), True), True)
        ])

        string_map_keys_schema = StructType(
            [StructField('map_field', MapType(StringType(), StringType()), nullable=True)])

        valid_schemas = [
            message_schema, prompt_response_schema, combined_schema, string_map_keys_schema
        ]

        schema_with_binary = StructType([StructField('data', BinaryType(), nullable=True)])

        # Schema with MapType having non-string keys
        non_string_map_keys_schema = StructType(
            [StructField('map_field', MapType(BinaryType(), StringType()), nullable=True)])

        # Schema with DateType and TimestampType
        schema_with_date_and_timestamp = StructType([
            StructField('birth_date', DateType(), nullable=True),
            StructField('event_timestamp', TimestampType(), nullable=True)
        ])

        invalid_schemas = [
            schema_with_binary, non_string_map_keys_schema, schema_with_date_and_timestamp
        ]

        for s in valid_schemas:
            assert is_json_compatible(s), str(s)

        for s in invalid_schemas:
            assert not is_json_compatible(s), str(s)

    def test_complex_schema(self,
                            complex_dataframe: Any,
                            local_remote_dir: tuple[str, str],
                            keep_local: bool = True,
                            merge_index: bool = True):
        out, _ = local_remote_dir
        mds_kwargs = {
            'out': out,
            'keep_local': keep_local,
            'columns': {
                'messages': 'json',
            },
        }

        def udf_iterable(df: Any):
            records = df.to_dict('records')
            for sample in records:
                v = list(sample)
                yield {'messages': v}

        _ = dataframe_to_mds(
            complex_dataframe,
            merge_index=merge_index,
            mds_kwargs=mds_kwargs,
            udf_iterable=udf_iterable,
            udf_kwargs=None,
        )
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
