# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A utility to convert spark dataframe to MDS."""

import logging
import os
import shutil
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

from streaming.base.util import get_import_exception_message
from streaming.base.util import merge_index as do_merge_index

try:
    from pyspark import TaskContext
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.types import (ArrayType, BinaryType, BooleanType, ByteType, DateType,
                                   DayTimeIntervalType, DecimalType, DoubleType, FloatType,
                                   IntegerType, LongType, NullType, ShortType, StringType,
                                   StructField, StructType, TimestampNTZType, TimestampType)
except ImportError as e:
    e.msg = get_import_exception_message(e.name, extra_deps='spark')  # pyright: ignore
    raise e

from streaming import MDSWriter
from streaming.base.format.index import get_index_basename
from streaming.base.format.mds.encodings import _encodings
from streaming.base.storage.upload import CloudUploader

logger = logging.getLogger(__name__)

SPARK_TO_MDS = {
    ByteType(): 'uint8',
    ShortType(): 'uint16',
    IntegerType(): 'int32',
    LongType(): 'int64',
    FloatType(): 'float32',
    DoubleType(): 'float64',
    DecimalType(): 'str_decimal',
    StringType(): 'str',
    BinaryType(): 'bytes',
    BooleanType(): None,
    TimestampType(): None,
    TimestampNTZType(): None,
    DateType(): None,
    DayTimeIntervalType(): None,
    ArrayType(IntegerType()): 'ndarray:int32',
    ArrayType(ShortType()): 'ndarray:int16',
    ArrayType(LongType()): 'ndarray:int64',
    ArrayType(FloatType()): 'ndarray:float32',
    ArrayType(DoubleType()): 'ndarray:float64',
}


def is_json_compatible(data_type: Any):
    """Recursively check if a given PySpark DataType is JSON compatible.

    JSON = Union[Dict[str, 'JSON'], List['JSON'], str, float, int, bool, None]

    Args:
        data_type (Any): A pyspark schema for a column of the input spark dataframe.

    Returns:
        (bool): True if data_type is JSON compatible.
    """
    if isinstance(data_type, StructType):
        return all(is_json_compatible(field.dataType) for field in data_type.fields)
    elif isinstance(data_type, ArrayType):
        return is_json_compatible(data_type.elementType)
    elif isinstance(data_type, (StringType, IntegerType, FloatType, BooleanType, NullType)):
        return True
    else:
        return False


def infer_dataframe_schema(dataframe: DataFrame,
                           user_defined_cols: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    """Retrieve schema to construct a dictionary or do sanity check for dataframe_to_mds.

    Args:
        dataframe (spark dataframe): dataframe to inspect schema
        user_defined_cols (Optional[Dict[str, Any]]): user specified schema for dataframe_to_mds

    Returns:
        If user_defined_cols is None, return schema_dict (dict): column name and dtypes that are
        supported by MDSWriter, else None

    Raises:
        ValueError if any of the datatypes are unsupported by dataframe_to_mds.
    """

    def map_spark_dtype(spark_data_type: Any) -> str:
        """Map spark data type to mds supported types.

        Args:
            spark_data_type: https://spark.apache.org/docs/latest/sql-ref-datatypes.html

        Returns:
            str: corresponding mds datatype for input.

        Raises:
            raise ValueError if no mds datatype is found for input type
        """
        if issubclass(type(spark_data_type), DecimalType):
            mds_type = SPARK_TO_MDS.get(DecimalType(), None)
        else:
            mds_type = SPARK_TO_MDS.get(spark_data_type, None)

        if mds_type is None:
            raise ValueError(f'{spark_data_type} is not supported by dataframe_to_mds')
        return mds_type

    # user has provided schema, we just check if mds supports the dtype
    if user_defined_cols is not None:
        mds_supported_dtypes = set(filter(bool, SPARK_TO_MDS.values()))

        for col_name, user_dtype in user_defined_cols.items():
            if col_name not in dataframe.columns:
                raise ValueError(
                    f'{col_name} is not a column of input dataframe: {dataframe.columns}')

            if user_dtype.startswith('ndarray:'):
                parts = user_dtype.split(':')
                if len(parts) == 3:
                    user_dtype = ':'.join(parts[:-1])

            actual_spark_dtype = dataframe.schema[col_name].dataType

            if user_dtype not in mds_supported_dtypes:
                if user_dtype == 'json':
                    if is_json_compatible(actual_spark_dtype):
                        continue
                    else:
                        raise ValueError(f'{col_name} can not be encoded by MDS JSON.')
                raise ValueError(f'{user_dtype} is not supported by dataframe_to_mds')

            mapped_mds_dtype = map_spark_dtype(actual_spark_dtype)
            if user_dtype != mapped_mds_dtype:
                raise ValueError(
                    f'Mismatched types: column name `{col_name}` is `{mapped_mds_dtype}` in ' +
                    f'DataFrame but `{user_dtype}` in user_defined_cols')
        return None

    schema = dataframe.schema
    schema_dict = {}

    for field in schema:
        dtype = map_spark_dtype(field.dataType)
        if dtype.split(':')[0] in _encodings:
            schema_dict[field.name] = dtype
        else:
            raise ValueError(f'{dtype} is not supported by dataframe_to_mds')
    return schema_dict


def dataframeToMDS(dataframe: DataFrame,
                   merge_index: bool = True,
                   mds_kwargs: Optional[Dict[str, Any]] = None,
                   udf_iterable: Optional[Callable] = None,
                   udf_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Deprecated API Signature.

    To be replaced by dataframe_to_mds
    """
    logger.warning(
        'The DataframeToMDS signature has been deprecated and will be removed in Streaming 0.8. ' +
        'Use dataframe_to_mds with the same arguments going forward')
    return dataframe_to_mds(dataframe, merge_index, mds_kwargs, udf_iterable, udf_kwargs)


def dataframe_to_mds(dataframe: DataFrame,
                     merge_index: bool = True,
                     mds_kwargs: Optional[Dict[str, Any]] = None,
                     udf_iterable: Optional[Callable] = None,
                     udf_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Execute a spark dataframe to MDS conversion process.

    This method orchestrates the conversion of a spark dataframe into MDS format by processing the
    input data, applying a user-defined iterable function if provided, and writing the results to
    an MDS-compatible format. The converted data is saved to mds_path.

    Args:
        dataframe (pyspark.sql.DataFrame): A DataFrame containing Delta Lake data.
        merge_index (bool): Whether to merge MDS index files. Defaults to ``True``.
        mds_kwargs (dict): Refer to https://docs.mosaicml.com/projects/streaming/en/stable/
            api_reference/generated/streaming.MDSWriter.html
        udf_iterable (Callable or None): A user-defined function that returns an iterable over the
            dataframe. udf_kwargs is the k-v args for the method. Defaults to ``None``.
        udf_kwargs (Dict): Additional keyword arguments to pass to the pandas processing
            function if provided. Defaults to an empty dictionary.

    Returns:
        mds_path (str or (str,str)): actual local and remote path were used
    Notes:
        - The method creates a SparkSession if not already available.
        - The 'udf_kwargs' dictionaries can be used to pass additional
          keyword arguments to the udf_iterable.
        - If udf_iterable is set, schema check will be skipped because the user defined iterable
          can create new columns. User must make sure they provide correct mds_kwargs[columns]
    """

    def write_mds(iterator: Iterable):
        """Worker node writes iterable to MDS datasets locally."""
        context = TaskContext.get()

        if context is not None:
            id = context.taskAttemptId()
        else:
            raise RuntimeError('TaskContext.get() returns None')

        if mds_path[1] == '':  # only local
            output = os.path.join(mds_path[0], f'{id}')
            partition_path = (output, '')
        else:
            output = (os.path.join(mds_path[0], f'{id}'), os.path.join(mds_path[1], f'{id}'))
            partition_path = output

        if mds_kwargs:
            kwargs = mds_kwargs.copy()
            kwargs['out'] = output
        else:
            kwargs = {}

        if merge_index:
            kwargs['keep_local'] = True  # need to keep workers' locals to do merge

        with MDSWriter(**kwargs) as mds_writer:
            for pdf in iterator:
                if udf_iterable is not None:
                    records = udf_iterable(pdf, **udf_kwargs or {})
                else:
                    records = pdf.to_dict('records')
                assert isinstance(
                    records,
                    Iterable), (f'pandas_processing_fn needs to return an iterable instead of a ' +
                                f'{type(records)}')

                for sample in records:
                    mds_writer.write(sample)

        yield pd.concat([
            pd.Series([os.path.join(partition_path[0], get_index_basename())],
                      name='mds_path_local'),
            pd.Series([
                os.path.join(partition_path[1], get_index_basename())
                if partition_path[1] != '' else ''
            ],
                      name='mds_path_remote')
        ],
                        axis=1)

    if dataframe is None or dataframe.isEmpty():
        raise ValueError(f'Input dataframe is None or Empty!')

    if not mds_kwargs:
        mds_kwargs = {}

    if not udf_kwargs:
        udf_kwargs = {}

    if 'out' not in mds_kwargs:
        raise ValueError(f'`out` and `columns` need to be specified in `mds_kwargs`')

    if udf_iterable is not None:
        if 'columns' not in mds_kwargs:
            raise ValueError(
                f'If udf_iterable is specified, user must provide correct `columns` in the ' +
                f'mds_kwargs')
        logger.warning("With udf_iterable defined, it's up to the user's discretion to provide " +
                       "mds_kwargs[columns]'")
    else:
        if 'columns' not in mds_kwargs:
            logger.warning(
                "User's discretion required: columns arg is missing from mds_kwargs. Will be " +
                'auto-inferred')
            mds_kwargs['columns'] = infer_dataframe_schema(dataframe)
            logger.warning(f"Auto inferred schema: {mds_kwargs['columns']}")
        else:
            infer_dataframe_schema(dataframe, mds_kwargs['columns'])

    out = mds_kwargs['out']
    keep_local = False if 'keep_local' not in mds_kwargs else mds_kwargs['keep_local']
    cu = CloudUploader.get(out, keep_local=keep_local)

    # Fix output format as mds_path: Tuple(local, remote)
    if cu.remote is None:
        mds_path = (cu.local, '')
    else:
        mds_path = (cu.local, cu.remote)

    # Prepare partition schema
    result_schema = StructType([
        StructField('mds_path_local', StringType(), False),
        StructField('mds_path_remote', StringType(), False),
    ])
    partitions = dataframe.mapInPandas(func=write_mds, schema=result_schema).collect()

    keep_local_files = True
    # If there are no remote part, we always keep the local
    # In case user forgot to set keep_local and set out to be a local path
    if cu.remote is not None:  # If there are no remote
        if 'keep_local' in mds_kwargs and mds_kwargs['keep_local'] == False:
            keep_local_files = False

    if merge_index:
        index_files = [(row['mds_path_local'], row['mds_path_remote']) for row in partitions]
        do_merge_index(index_files, out, keep_local=keep_local_files, download_timeout=60)

    if not keep_local_files:
        shutil.rmtree(cu.local, ignore_errors=True)

    return mds_path
