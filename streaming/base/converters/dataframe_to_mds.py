# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A utility to convert spark dataframe to MDS."""

import json
import logging
import os
import shutil
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import pandas as pd

from streaming.base.util import get_import_exception_message

try:
    from pyspark import TaskContext
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.types import (ArrayType, BinaryType, BooleanType, ByteType, DateType,
                                   DayTimeIntervalType, DecimalType, DoubleType, FloatType,
                                   IntegerType, LongType, MapType, ShortType, StringType,
                                   StructField, StructType, TimestampNTZType, TimestampType)
except ImportError as e:
    e.msg = get_import_exception_message(e.name, extra_deps='spark')  # pyright: ignore
    raise e

from streaming import MDSWriter
from streaming.base.format.index import get_index_basename
from streaming.base.format.mds.encodings import _encodings
from streaming.base.storage.upload import CloudUploader

logger = logging.getLogger(__name__)


def is_iterable(obj: object) -> bool:
    """Check if obj is iterable.

    Args:
        obj: python object
    Return:
        bool: true if obj is iterable false otherwise
    """
    return issubclass(type(obj), Iterable)


def infer_dataframe_schema(dataframe: DataFrame,
                           user_defined_cols: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    """Retrieve schema to construct a dictionary or do sanity check for MDSWriter.

    Args:
        dataframe (spark dataframe): dataframe to inspect schema
        user_defined_cols (Optional[Dict[str, Any]]): user specified schema for MDSWriter
    Return:
        If user_defined_cols is None, return schema_dict (dict): column name and dtypes that supported by MDSWriter
        Else, return None
    Exceptions:
        Any of the datatype found to be unsupported by MDSWriter, then raise ValueError
    """
    mapping_spark_to_mds = {
        ByteType: 'uint8',
        ShortType: 'uint16',
        IntegerType: 'int',
        LongType: 'int64',
        FloatType: 'float32',
        DoubleType: 'float64',
        DecimalType: 'str_decimal',
        StringType: 'str',
        BinaryType: 'bytes',
        BooleanType: None,
        TimestampType: None,
        TimestampNTZType: None,
        DateType: None,
        DayTimeIntervalType: None,
        ArrayType: None,
        MapType: None,
        StructType: None,
        StructField: None
    }

    def map_spark_dtype(spark_data_type: Any) -> str:
        mds_type = mapping_spark_to_mds.get(type(spark_data_type), None)
        if mds_type is None:
            raise ValueError(f'{spark_data_type} is not supported by MDSwriter')
        return mds_type

    if user_defined_cols is not None:  # user has provided schema, we just check if mds supports the dtype
        mds_supported_dtypes = {
            mds_type for mds_type in mapping_spark_to_mds.values() if mds_type is not None
        }
        for col_name, user_dtype in user_defined_cols.items():
            if col_name not in dataframe.columns:
                raise ValueError(
                    f'{col_name} is not a column of input dataframe: {dataframe.columns}')
            if user_dtype not in mds_supported_dtypes:
                raise ValueError(f'{user_dtype} is not supported by MDSwriter')

            actual_spark_dtype = dataframe.schema[col_name].dataType
            mapped_mds_dtype = map_spark_dtype(actual_spark_dtype)
            if user_dtype != mapped_mds_dtype:
                raise ValueError(
                    f'Mismatched types: column name `{col_name}` is `{mapped_mds_dtype}` in DataFrame but `{user_dtype}` in user_defined_cols'
                )
        return None

    schema = dataframe.schema
    schema_dict = {}

    for field in schema:
        dtype = map_spark_dtype(field.dataType)
        if dtype in _encodings:
            schema_dict[field.name] = dtype
        else:
            raise ValueError(f'{dtype} is not supported by MDSwriter')
    return schema_dict


def do_merge_index(partitions: Iterable, mds_path: Union[str, Tuple[str, str]]):
    """Merge index.json from partitions into one for streaming.

    Args:
        partitions (Iterable): partitions that contain pd.DataFrame
        mds_path (Union[str, Tuple[str, str]]): (str,str)=(local,remote), str = local or remote based on parse_uri(url) result
    Return:
        None
    """
    if not partitions:
        logger.warning('No partitions exist, no index merged')
        return

    shards = []

    for row in partitions:
        mds_partition_index = f'{row.mds_path}/{get_index_basename()}'
        mds_partition_basename = os.path.basename(row.mds_path)
        obj = json.load(open(mds_partition_index))
        for i in range(len(obj['shards'])):
            shard = obj['shards'][i]
            for key in ('raw_data', 'zip_data'):
                if shard.get(key):
                    basename = shard[key]['basename']
                    obj['shards'][i][key]['basename'] = os.path.join(mds_partition_basename,
                                                                     basename)
        shards += obj['shards']

    obj = {
        'version': 2,
        'shards': shards,
    }

    if isinstance(mds_path, str):
        mds_index = os.path.join(mds_path, get_index_basename())
    else:
        mds_index = os.path.join(mds_path[0], get_index_basename())

    with open(mds_index, 'w') as out:
        json.dump(obj, out)


def dataframeToMDS(dataframe: DataFrame,
                   merge_index: bool = True,
                   mds_kwargs: Optional[Dict[str, Any]] = None,
                   udf_iterable: Optional[Callable] = None,
                   udf_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Any, int]:
    """Execute a spark dataframe to MDS conversion process.

    This method orchestrates the conversion of a spark dataframe into MDS format by
    processing the input data, applying a user-defined iterable function if
    provided, and writing the results to MDS-compatible format. The converted data is saved to mds_path.

    Args:
        dataframe (pyspark.sql.DataFrame): A DataFrame containing Delta Lake data.
        merge_index (bool): Whether to merge MDS index files. Defaults to ``True``.
        mds_kwargs (dict): Refer to https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html
        udf_iterable (Callable or None): A user-defined function that returns an iterable over the dataframe. udf_kwargs is the k-v args for the method. Defaults to ``None``.
        udf_kwargs (Dict): Additional keyword arguments to pass to the pandas processing
            function if provided. Defaults to an empty dictionary.

    Return:
        mds_path (str or (str,str)): actual local and remote path were used
        fail_count (int): number of records failed to be converted

    Note:
        - The method creates a SparkSession if not already available.
        - The 'udf_kwargs' dictionaries can be used to pass additional
          keyword arguments to the udf_iterable.
        - If udf_iterable is set, schema check will be skipped because the user defined iterable can create new columns.
    """

    def write_mds(iterator: Iterable):

        context = TaskContext.get()

        if context is not None:
            id = context.taskAttemptId()
        else:
            raise RuntimeError('TaskContext.get() returns None')

        if isinstance(mds_path, str):  # local
            output = os.path.join(mds_path, f'{id}')
            out_file_path = output
        else:
            output = (os.path.join(mds_path[0], f'{id}'), os.path.join(mds_path[1], f'{id}'))
            out_file_path = output[0]

        if mds_kwargs:
            kwargs = mds_kwargs.copy()
            kwargs['out'] = output
        else:
            kwargs = {}

        if merge_index:
            kwargs['keep_local'] = True  # need to keep local to do merge

        count = 0

        with MDSWriter(**kwargs) as mds_writer:
            for pdf in iterator:
                if udf_iterable is not None:
                    records = udf_iterable(pdf, **udf_kwargs or {})
                else:
                    records = pdf.to_dict('records')
                assert is_iterable(
                    records
                ), f'pandas_processing_fn needs to return an iterable instead of a {type(records)}'

                for sample in records:
                    try:
                        mds_writer.write(sample)
                    except:
                        raise RuntimeError(f'failed to write sample: {sample}')
                        count += 1

        yield pd.concat(
            [pd.Series([out_file_path], name='mds_path'),
             pd.Series([count], name='fail_count')],
            axis=1)

    if dataframe is None or dataframe.count() == 0:
        raise ValueError(f'input dataframe is none or empty!')

    if not mds_kwargs:
        mds_kwargs = {}

    if not udf_kwargs:
        udf_kwargs = {}

    if 'out' not in mds_kwargs:
        raise ValueError(f'out and columns need to be specified in mds_kwargs')

    if udf_iterable is None:
        if 'columns' not in mds_kwargs:
            logger.warning(
                "User's discretion required: columns arg is missing from mds_kwargs. Will be auto inferred"
            )
            mds_kwargs['columns'] = infer_dataframe_schema(dataframe)
            logger.warning(f"Auto inferred schema: {mds_kwargs['columns']}")
        else:
            infer_dataframe_schema(dataframe, mds_kwargs['columns'])

    out = mds_kwargs['out']
    keep_local = False if 'keep_local' not in mds_kwargs else mds_kwargs['keep_local']
    cu = CloudUploader.get(out, keep_local=keep_local)

    # Fix output format as mds_path: Tuple => remote Str => local only
    if cu.remote is None:
        mds_path = cu.local
    else:
        mds_path = (cu.local, cu.remote)

    # Prepare partition schema
    result_schema = StructType([
        StructField('mds_path', StringType(), False),
        StructField('fail_count', IntegerType(), False)
    ])
    partitions = dataframe.mapInPandas(func=write_mds, schema=result_schema).collect()

    if merge_index:
        do_merge_index(partitions, mds_path)

    if cu.remote is not None:
        if merge_index:
            cu.upload_file(get_index_basename())
        if 'keep_local' in mds_kwargs and mds_kwargs['keep_local'] == False:
            shutil.rmtree(cu.local, ignore_errors=True)

    summ_fail_count = 0
    for row in partitions:
        summ_fail_count += row['fail_count']

    if summ_fail_count > 0:
        logger.warning(
            f'Total failed records = {summ_fail_count}\nOverall records {dataframe.count()}')
    return mds_path, summ_fail_count
