# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A utility to convert spark dataframe to MDS."""

import collections
import json
import logging
import os
import shutil
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import pandas as pd
from pyspark import TaskContext
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (ArrayType, BinaryType, BooleanType, ByteType, DateType,
                               DayTimeIntervalType, DecimalType, DoubleType, FloatType,
                               IntegerType, LongType, MapType, ShortType, StringType, StructField,
                               StructType, TimestampNTZType, TimestampType)

from streaming import MDSWriter
from streaming.base.format.index import get_index_basename
from streaming.base.format.mds.encodings import _encodings
from streaming.base.storage.upload import CloudUploader

logger = logging.getLogger(__name__)


def is_iterable(obj: Any) -> bool:
    """Check if obj is iterable."""
    return issubclass(type(obj), Iterable)


def infer_dataframe_schema(dataframe: DataFrame,
                           user_defined_cols: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    """Takes a pyspark dataframe and retrives the schema information, constructing a dictionary."""
    dtype_mapping = {
        ByteType: 'bytes',
        ShortType: 'uint64',
        IntegerType: 'int',
        LongType: 'int64',
        FloatType: 'float32',
        DoubleType: 'float64',
        DecimalType: None,
        StringType: 'str',
        BinaryType: None,
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

    def map_spark_dtype(spark_data_type: Any):
        for k, v in dtype_mapping.items():
            if isinstance(spark_data_type, k) and v is not None:
                return v
        raise ValueError(f'{spark_data_type} is not supported by MDSwrite')

    if user_defined_cols is not None:  # user has provided schema, we just check if mds supports the dtype
        mds_supported_dtypes = list(dtype_mapping.values())
        for k, v in user_defined_cols.items():
            if k not in dataframe.columns:
                raise ValueError(f'{k} is not a column of input dataframe')
            if v not in mds_supported_dtypes:
                raise ValueError(f'{v} is not supported by MDSwrite')
        return None

    schema = dataframe.schema
    schema_dict = {}

    for field in schema:
        dtype = map_spark_dtype(field.dataType)
        if dtype in _encodings:
            schema_dict[field.name] = dtype
        else:
            print(_encodings)

    return schema_dict


def do_merge_index(partitions: Iterable,
                   mds_path: Union[str, Tuple[str, str]],
                   skip: bool = False):
    """Merge index.json from partitions into one for streaming.

    Args:
        partitions (Iterable): partitions that contain pd.DataFrame
        mds_path (Tuple or str): (str,str)=(local,remote), str = local or remote based on parse_uri(url) result
        skip (bool): whether to merge index from partitions
    """
    if not partitions or skip:
        return

    shards = []

    for row in partitions:
        mds_partition_index = f'{row.mds_path}/{get_index_basename()}'
        mds_partition_basename = os.path.basename(row.mds_path)
        obj = json.load(open(mds_partition_index))
        for i in range(len(obj['shards'])):
            shard = obj['shards'][i]
            for key in ['raw_data', 'zip_data']:
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
                   sample_ratio: float = -1.0,
                   mds_kwargs: Optional[Dict[str, Any]] = None,
                   udf_iterable: Optional[Callable] = None,
                   udf_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Any, int]:
    """Execute a spark dataframe to MDS conversion process.

    This method orchestrates the conversion of a spark dataframe into MDS format by
    processing the input data, applying a user-defined iterable function if
    provided, and writing the results to MDS-compatible format. The converted data is saved to mds_path.

    Args:
        dataframe (pyspark.sql.DataFrame or None): A DataFrame containing Delta Lake data.
        merge_index (bool): Whether to merge MDS index files. Default is True.
        sample_ratio (float): The fraction of data to randomly sample during conversion.
            Should be in the range (0, 1). Default is -1.0 (no sampling).

        mds_kwargs (dict): arguments for MDSwrite.
        out (str | Tuple[str, str]): Output dataset directory to save shard files.
            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in the
               `local_dir` and also uploaded to a remote location.
        columns (Dict[str, str]): Sample columns. If not specified, use all cols and inferred dtype from dataframe.
        keep_local (bool): If the dataset is uploaded, whether to keep the local dataset directory
            or remove it after uploading. Defaults to ``False``.
        compression (str, optional): Optional compression or compression:level. Defaults to
            ``None``.
        hashes (List[str], optional): Optional list of hash algorithms to apply to shard files.
            Defaults to ``None``.
        size_limit (Union[int, str], optional): Optional shard size limit, after which point to start a new
            shard. If ``None``, puts everything in one shard. Can specify bytes
            human-readable format as well, for example ``"100kb"`` for 100 kilobyte
            (100*1024) and so on. Defaults to ``1 << 26``
        udf_iterable (Callable or None): A user-defined function that returns an iterable over the dataframe. ppfn_kwargs is the k-v args for the method. Default is None.
        udf_kwargs (Dict): Additional keyword arguments to pass to the pandas processing
            function if provided. Default is an empty dictionary.

    Return:
        mds_path: actual local and remote path were used
        fail_count: number of records failed to be converted
    Raises:
        ValueError: If dataframe is not provided

    Note:
        - The method creates a SparkSession if not already available.
        - If 'sample_ratio' is provided, the input data will be randomly sampled.
        - The 'ppfn_kwargs' dictionaries can be used to pass additional
          keyword arguments to the udf_iterable.
    """

    def write_mds(iterator: Iterable):

        context = TaskContext.get()
        if context is not None:
            id = context.taskAttemptId()
        else:
            raise Exception('TaskContext.get() returns None')

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
                        logger.warning(f'failed to write sample: {sample}')
                        count += 1

        yield pd.concat(
            [pd.Series([out_file_path], name='mds_path'),
             pd.Series([count], name='fail_count')],
            axis=1)

    if dataframe is None or dataframe.count() == 0:
        raise ValueError(f'input dataframe is none or empty!')

    if not mds_kwargs:
        mds_kwargs = collections.defaultdict(str)

    if not udf_kwargs:
        udf_kwargs = collections.defaultdict(str)

    if 'out' not in mds_kwargs:
        raise ValueError(f'out and columns need to be specified in mds_kwargs')

    if 'columns' not in mds_kwargs:
        logger.warning(
            "User's discretion required: columns arg is missing from mds_kwargs. Will be auto inferred"
        )
        mds_kwargs['columns'] = infer_dataframe_schema(dataframe)
    else:
        infer_dataframe_schema(dataframe, mds_kwargs['columns'])

    if 0 < sample_ratio < 1:
        dataframe = dataframe.sample(sample_ratio)

    out = mds_kwargs['out']
    keep_local = False if 'keep_local' not in mds_kwargs else mds_kwargs['keep_local']
    cu = CloudUploader.get(out, keep_local=keep_local)
    if os.path.exists(cu.local) and len(os.listdir(cu.local)) != 0:
        raise ValueError(
            'Looks like {out} is local folder and it is not empty. MDSwriter needs an empty local folder to proceed.'
        )
        return

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

    do_merge_index(partitions, mds_path, skip=not merge_index)

    if cu.remote is not None:
        if merge_index == True:
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


if __name__ == '__main__':

    spark = SparkSession.builder.getOrCreate()  # pyright: ignore

    def parse_args() -> Namespace:
        """Parse commandline arguments."""
        parser = ArgumentParser(
            description=
            'Convert dataset into MDS format. Running from command line does not support optionally processing functions!'
        )
        parser.add_argument('--delta_table_path', type=str, required=True)
        parser.add_argument('--mds_path', type=str, required=True)
        parser.add_argument('--partition_size', type=int, required=True)
        parser.add_argument('--merge_index', type=bool, required=True)

        parsed = parser.parse_args()
        return parsed

    args = parse_args()

    df = spark.read.table(args.delta_table_path)
    dataframeToMDS(df,
                   mds_kwargs={
                       'out': args.mds_path,
                       'columns': {
                           'tokens': 'bytes'
                       }
                   },
                   merge_index=args.merge_index,
                   sample_ratio=args.sample_ratio)
