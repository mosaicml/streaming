# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from typing import List, Optional, Tuple, Dict, Any

import pytest

from streaming.base.constant import RESUME
from streaming.base.shared.prefix import _get_path
from streaming.base.util import (bytes_to_int, clean_stale_shared_memory, get_list_arg,
                                 number_abbrev_to_int)
from tests.common.utils import convert_to_mds

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

from streaming.base.converters import dataframeToMDS
from streaming.base.util import merge_index
import tempfile
import os
import glob

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


@pytest.fixture
def dataframe():
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

"""Example input urls to test
    ['gs://mybucket/mdsdata/25/'...]
    ['/path/never/exists/25',... ]
    [('/path/never/exists/25', 'gs://mybucket/mdsdata/25/'), ...]
    [('tests/resources/naive_MDSdataset/25/', 'gs://mybucket/mdsdata/25/'), ...]
"""
@pytest.mark.parametrize('folder_urls', ['local_accessible', 'remote', 'local_unaccessible', 'local_accessible_tuple'])
@pytest.mark.parametrize('out', ['local_str', 'remote_str', 'tuple'])
@pytest.mark.usefixtures('local_remote_dir')
@pytest.mark.parametrize('keep_local', [True, False])
def test_merge_index(local_remote_dir: Tuple[str, str],
                     dataframe: Any,
                     keep_local: bool,
                     folder_urls: Dict,
                     out: Dict):

    naive_mds_partitions= ['tests/resources/naive_MDSdataset/25/',
                           'tests/resources/naive_MDSdataset/26/',
                           'tests/resources/naive_MDSdataset/27/']

    if folder_urls == 'local_accessible':
        folder_urls = [ os.getcwd() + '/' + s for s in naive_mds_partitions]
        print(folder_urls)

        if out == 'local_str':
            with tempfile.TemporaryDirectory() as tmp:
                n_downloads = merge_index(folder_urls, tmp, keep_local = keep_local, overwrite = True)
                if keep_local:
                    assert(os.path.exists(os.path.join(tmp, 'index.json')))
                else:
                    assert(not os.path.exists(os.path.join(tmp, 'index.json')))
                assert n_downloads == 0, f"n_downloads should be 0 instead of {n_downloads}"
        else:
            return

    if folder_urls == 'remote':
        return

    if folder_urls == 'local_unaccessible':
        with tempfile.TemporaryDirectory() as tmp_data_root:
            folder_urls = [ tmp_data_root + '/' + s for s in naive_mds_partitions]
            with pytest.raises(FileNotFoundError, match=f'.* does not exit or cannot be acceessed by the current process.*'):
                merge_index(folder_urls, tmp_data_root, keep_local = keep_local, overwrite = True)

    if folder_urls == 'local_accessible_tuple':
        folder_urls = []
        for s in naive_mds_partitions:
            folder_urls.append((os.getcwd() + '/' + s, 'gs://mybucket/'+ s ))
        if out == 'local_str':
            with tempfile.TemporaryDirectory() as tmp_data_root:
                folder_urls = [ tmp_data_root + '/' + s for s in naive_mds_partitions]
                with pytest.raises(FileNotFoundError, match=f'.* does not exit or cannot be acceessed by the current process.*'):
                    merge_index(folder_urls, tmp_data_root, keep_local = keep_local, overwrite = True)
        else:
            return


