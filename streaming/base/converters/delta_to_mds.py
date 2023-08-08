# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0
"""A utility to convert databricks' tables to MDS."""


import json
import os
import shutil
import uuid
import warnings
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterable

import mlflow
import pandas as pd
from pyspark import TaskContext
from pyspark.sql.types import StringType, StructField, StructType

from streaming import MDSWriter

default_mds_kwargs = {
    'compression': 'zstd:7',
    'hashes': ['sha1', 'xxh64'],
    'size_limit': 1 << 27,
    'progress_bar': 1,
    'columns': {
        'tokens': 'bytes'
    },
}

default_ppfn_kwargs = {
    'concat_tokens': 2048,
    'tokenizer': 'EleutherAI/gpt-neox-20b',
    'eos_text': '<|endoftext|>',
    'compression': 'zstd',
    'split': 'train',
    'no_wrap': False,
    'bos_text': '',
    'key': 'content',
}


def is_iterable(obj):
    """Check if obj is iterable"""
    return issubclass(type(obj), Iterable)


def parse_args():
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


class DeltaMdsConverter(mlflow.pyfunc.PythonModel):
    """A class for converting Delta Lake data into MDS format using PySpark and pandas.

    This class provides methods to convert Delta Lake data into MDS (Model Deployment
    Service) format, which is optimized for efficient model serving. The conversion
    process involves processing the input data using a user-defined pandas processing
    function and writing the results to MDS-compatible format.

    Args:
        mlflow.pyfunc.PythonModel: A base class for defining Python-based MLflow models.

    Methods:
        spark_jobs(self, proc_fn, ppfn_kwargs: Dict = {}, mds_kwargs: Dict = {}):
            Converts the Delta Lake data into MDS format using PySpark and pandas.

        execute(self, dataframe=None, delta_parquet_path: str = '', delta_table_path: str = '',
                mds_path: str = '', partition_size: int = 1, merge_index: bool = True,
                pandas_processing_fn: Callable = None, sample_ratio: float = -1.0,
                remote: str = '', overwrite: bool = True, mds_kwargs: Dict = {},
                ppfn_kwargs: Dict = {}):
            Executes the Delta Lake to MDS conversion process.

    Attributes:
        spark: A SparkSession instance for managing Spark applications.
        df_delta: A DataFrame containing the Delta Lake data.
        result_schema: A schema for the results containing a single 'mds_path' column.
        partition_size: The number of partitions to use during conversion.
        merge_index: A boolean indicating whether to merge MDS index files.
    """

    def spark_jobs(self, proc_fn, ppfn_kwargs: Dict = {}, mds_kwargs: Dict = {}):

        def write_mds(iterator):

            id = TaskContext.get().taskAttemptId()
            out_file_path = os.path.join(mds_kwargs['out'], f'{id}')
            mds_kwargs.pop('out')

            with MDSWriter(out=out_file_path, **mds_kwargs) as mds_writer:
                for pdf in iterator:
                    if proc_fn is not None:
                        d = proc_fn(pdf, **ppfn_kwargs)
                    else:
                        d = pdf.to_dict('records')
                    assert is_iterable(
                        d
                    ), f'pandas_processing_fn needs to return an iterable instead of a {type(d)}'

                    for row in d:
                        mds_writer.write(row)
            yield pd.DataFrame(pd.Series([out_file_path], name='mds_path'))

        def merge_index(partitions):
            shards = []

            for row in partitions:
                mds_partition_index = f'{row.mds_path}/index.json'
                mds_partition_basename = os.path.basename(row.mds_path)
                obj = json.load(open(mds_partition_index))
                for i in range(len(obj['shards'])):
                    shard = obj['shards'][i]
                    for key in ['raw_data', 'zip_data']:
                        if shard.get(key):
                            basename = shard[key]['basename']
                            obj['shards'][i][key]['basename'] = os.path.join(
                                mds_partition_basename, basename)
                shards += obj['shards']

            obj = {
                'version': 2,
                'shards': shards,
            }

            mds_index = os.path.join(mds_kwargs['out'], 'index.json')

            with open(mds_index, 'w') as out:
                json.dump(obj, out)

        if self.partition_size > 0:
            partitions = self.df_delta.repartition(self.partition_size).mapInPandas(
                func=write_mds, schema=self.result_schema).collect()
        else:
            partitions = self.df_delta.mapInPandas(func=write_mds,
                                                   schema=self.result_schema).collect()

        if self.merge_index:
            merge_index(partitions)

    def execute(self,
                dataframe=None,
                delta_parquet_path: str = '',
                delta_table_path: str = '',
                mds_path: str = '',
                partition_size: int = 1,
                merge_index: bool = True,
                pandas_processing_fn: Callable = None,
                sample_ratio: float = -1.0,
                remote: str = '',
                overwrite: bool = True,
                mds_kwargs: Dict = {},
                ppfn_kwargs: Dict = {}):
        """Execute the Delta Lake to MDS conversion process.

        This method orchestrates the conversion of Delta Lake data into MDS format by
        processing the input data, applying a user-defined pandas processing function if
        provided, and writing the results to MDS-compatible format. The converted data is
        saved to the specified 'mds_path' location.

        Args:
            dataframe (pyspark.sql.DataFrame or None): A DataFrame containing Delta Lake data.
                If provided, this DataFrame will be used for conversion.
            delta_parquet_path (str): The path to the Delta Lake data in Parquet format.
            delta_table_path (str): The path to the Delta Lake data as a SQL table.
            mds_path (str): The path where the converted MDS data will be stored.
            partition_size (int): The number of partitions to use during conversion. Default is 1.
            merge_index (bool): Whether to merge MDS index files. Default is True.
            pandas_processing_fn (Callable or None): A user-defined pandas processing function
                to apply to the input data before conversion. Default is None.
            sample_ratio (float): The fraction of data to randomly sample during conversion.
                Should be in the range (0, 1). Default is -1.0 (no sampling).
            remote (str): The remote location type (e.g., 'dbfs') if using a remote path.
                Default is an empty string.
            overwrite (bool): Whether to overwrite the existing 'mds_path' folder if it exists.
                Default is True.
            mds_kwargs (Dict): Additional keyword arguments to pass to the MDSWriter class
                during conversion. Default is an empty dictionary.
            ppfn_kwargs (Dict): Additional keyword arguments to pass to the pandas processing
                function if provided. Default is an empty dictionary.

        Returns:
            None

        Raises:
            ValueError: If both 'delta_parquet_path' and 'delta_table_path' cannot be read.

        Note:
            - The method creates a SparkSession if not already available.
            - If 'dataframe' is provided, it takes precedence over 'delta_parquet_path'
              and 'delta_table_path'.
            - If 'sample_ratio' is provided, the input data will be randomly sampled.
            - The 'remote' argument can be used to specify different remote storage types.
            - The 'mds_kwargs' and 'ppfn_kwargs' dictionaries can be used to pass additional
              keyword arguments to the MDSWriter and pandas processing function, respectively.
        """
        import pyspark
        self.spark = pyspark.sql.SparkSession.builder.getOrCreate()

        if dataframe is not None:
            self.df_delta = dataframe
        else:
            try:
                self.df_delta = self.spark.read.parquet(delta_parquet_path)
            except:
                try:
                    self.df_delta = self.spark.read.table(delta_table_path)
                except:
                    raise ValueError(
                        f'Both input tables: {delta_parquet_path}, {delta_table_path} cannot be read!'
                    )

        # Prepare partition schema
        self.result_schema = StructType([StructField('mds_path', StringType(), False)])

        if 0 < sample_ratio < 1:
            self.df_delta = self.df_delta.sample(sample_ratio)

        # Clean up dest folder
        mnt_path = mds_path

        if remote != '':
            assert (remote == 'dbfs'), 'Other remotes are not developed yet'
            mnt_path = f'/{remote}/{mds_path}'

        if not overwrite:
            try:
                shutil.rmtree(mnt_path)
                os.makedirs(mnt_path)
            except:
                print(
                    'Ignore for now rmtree and os.makedirs error: folder exists permission issue etc.'
                )

        mds_kwargs['out'] = mnt_path

        # Set internal variables
        self.partition_size = partition_size
        self.merge_index = merge_index

        # Start spark job and log artifacts with mlflow
        self.spark_jobs(pandas_processing_fn, ppfn_kwargs, mds_kwargs)

        with mlflow.start_run() as run:

            # mlflow log
            #model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=self)
            mlflow.log_param('delta_parquet_path', delta_parquet_path)
            mlflow.log_param('delta_table_path', delta_table_path)
            mlflow.log_param('mds_path', mds_path)
            mlflow.log_param('remote', remote)
            mlflow.log_param('pandas_processing_fn', pandas_processing_fn)
            mlflow.log_param('partition_size', partition_size)
            mlflow.log_param('merge_index', merge_index)
            mlflow.log_param('sample_ratio', sample_ratio)
            mlflow.log_param('overwrite', overwrite)
            dataset = mlflow.data.from_spark(dataframe)
            mlflow.log_dict(default_mds_kwargs, 'default_mds_kwargs.json')
            mlflow.log_dict(default_ppfn_kwargs, 'default_ppfn_kwargs.json')


def test():
    """test from databricks."""
    dmc = DeltaMdsConverter()

    default_ppfn_kwargs.pop('key')

    remote = ''
    input_path = '/refinedweb/raw'
    mds_path = '/Volumes/datasets/default/mosaic_hackathon/mds/ml/refinedweb'

    dmc.execute(delta_parquet_path=input_path,
                mds_path=mds_path,
                partition_size=2048,
                merge_index=True,
                pandas_processing_fn=pandas_processing_fn,
                sample_ratio=-1,
                remote=remote,
                mds_kwargs=default_mds_kwargs,
                ppfn_kwargs=default_ppfn_kwargs)


if __name__ == '__main__':

    args = parse_args()

    dmc = DeltaMdsConverter()

    dmc.execute(delta_parquet_path=args.delta_parquet_path,
                delta_table_path=args.delta_table_path,
                mds_path=args.mds_path,
                partition_size=args.partition_size,
                merge_index=args.merge_index,
                pandas_processing_fn=None,
                sample_ratio=args.sample_ratio,
                mds_kwargs=default_mds_kwargs,
                ppfn_kwargs=default_ppfn_kwargs)
