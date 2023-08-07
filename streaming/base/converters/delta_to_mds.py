import os
import json
import warnings
from typing import Dict, Iterable, Callable, Any
from argparse import ArgumentParser, Namespace
import uuid
from streaming import MDSWriter
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType
from pyspark import TaskContext
import mlflow
from collections.abc import Iterable
import shutil


default_mds_kwargs = {
    'compression': 'zstd:7',
    'hashes': ['sha1','xxh64'],
    'size_limit': 1<<27,
    'progress_bar':1,
    'columns':{'tokens': 'bytes'},
}

default_ppfn_kwargs = {
    'concat_tokens' : 2048,
    'tokenizer' : "EleutherAI/gpt-neox-20b",
    'eos_text' : '<|endoftext|>',
    'compression' : "zstd",
    'split' : 'train',
    'no_wrap' : False,
    'bos_text' : '',
    'key' : 'content',
}

def is_iterable(obj):
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

    def spark_jobs(self,
                   proc_fn,
                   ppfn_kwargs: Dict = {},
                   mds_kwargs: Dict = {}):


        def write_mds(iterator):

            id = TaskContext.get().taskAttemptId()
            out_file_path = os.path.join(mds_kwargs["out"], f'{id}')
            mds_kwargs.pop('out')

            with MDSWriter(out=out_file_path, **mds_kwargs) as mds_writer:
                for pdf in iterator:
                    if proc_fn is not None:
                        d = proc_fn(pdf, **ppfn_kwargs)
                    else:
                        d = pdf.to_dict('records')
                    assert is_iterable(d), f"pandas_processing_fn needs to return an iterable instead of a {type(d)}"

                    for row in d:
                        mds_writer.write(row)
            yield pd.DataFrame(pd.Series([out_file_path], name="mds_path"))

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
                            obj['shards'][i][key]['basename'] = os.path.join(mds_partition_basename, basename)
                shards += obj['shards']

            obj = {
              'version': 2,
              'shards': shards,
            }

            mds_index = os.path.join(mds_kwargs["out"], 'index.json')

            with open(mds_index, 'w') as out:
                json.dump(obj, out)

        if self.partition_size > 0:
            partitions = self.df_delta.repartition(self.partition_size).mapInPandas(func=write_mds, schema=self.result_schema).collect()
        else:
            partitions = self.df_delta.mapInPandas(func=write_mds, schema=self.result_schema).collect()

        if self.merge_index:
            merge_index(partitions)


    def execute(self,
                dataframe = None,
                delta_parquet_path : str = '',
                delta_table_path : str = '',
                mds_path : str = '',
                partition_size : int = 1,
                merge_index : bool = True,
                pandas_processing_fn : Callable = None,
                sample_ratio : float = -1.0,
                remote : str = '',
                overwrite : bool = True,
                mds_kwargs: Dict = {},
                ppfn_kwargs: Dict = {}):

        # Read data

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
                    raise ValueError(f"Both input tables: {delta_parquet_path}, {delta_table_path} cannot be read!")

        # Prepare partition schema
        self.result_schema = StructType([StructField("mds_path", StringType(), False)])

        if 0 < sample_ratio < 1:
            self.df_delta = self.df_delta.sample(sample_ratio)

        # Clean up dest folder
        mnt_path = mds_path

        if remote != '':
            assert(remote == 'dbfs'), "Other remotes are not developed yet"
            mnt_path = f'/{remote}/{mds_path}'

        if not overwrite:
            try:
                shutil.rmtree(mnt_path)
                os.makedirs(mnt_path)
            except:
                print('Ignore for now rmtree and os.makedirs error: folder exists permission issue etc.')

        mds_kwargs['out'] = mnt_path

        # Set internal variables
        self.partition_size = partition_size
        self.merge_index = merge_index

        # Start spark job and log artifacts with mlflow
        self.spark_jobs(pandas_processing_fn, ppfn_kwargs, mds_kwargs)


        with mlflow.start_run() as run:

            # mlflow log
            model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=self)
            mlflow.log_param("delta_parquet_path", delta_parquet_path)
            mlflow.log_param("delta_table_path", delta_table_path)
            mlflow.log_param("mds_path", mds_path)
            mlflow.log_param("remote", remote)
            mlflow.log_param("pandas_processing_fn", pandas_prcessing_fn)
            mlflow.log_param("partition_size", partition_size)
            mlflow.log_param("merge_index", merge_index)
            mlflow.log_param("sample_ratio", sample_ratio)
            mlflow.log_param("overwrite", overwrite)
            dataset =mlflow.data.from_spark(dataframe)
            mlflow.log_dict(default_mds_kwargs, 'default_mds_kwargs.json')
            mlflow.log_dict(default_ppfn_kwargs, 'default_ppfn_kwargs.json')

            model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=dmc)
            mlflow.log_dict(default_mds_kwargs, 'default_mds_kwargs.json')
            mlflow.log_dict(default_ppfn_kwargs, 'default_ppfn_kwargs.json')

def test():

    dmc = DeltaMdsConverter()

    default_ppfn_kwargs.pop('key')

    remote = ''
    input_path =  '/refinedweb/raw'
    mds_path =  '/Volumes/datasets/default/mosaic_hackathon/mds/ml/refinedweb'

    dmc.execute(delta_parquet_path = input_path,
                mds_path = mds_path,
                partition_size = 2048,
                merge_index = True,
                pandas_processing_fn = pandas_processing_fn,
                sample_ratio = -1,
                remote = remote,
                mds_kwargs = default_mds_kwargs,
                ppfn_kwargs = default_ppfn_kwargs)


if __name__ == "__main__":

    args = parse_args()

    dmc = DeltaMdsConverter()

    dmc.execute(delta_parquet_path = args.delta_parquet_path,
                delta_table_path = args.delta_table_path,
                mds_path = args.mds_path,
                partition_size = args.partition_size,
                merge_index = args.merge_index,
                pandas_processing_fn = None,
                sample_ratio = args.sample_ratio,
                mds_kwargs = default_mds_kwargs,
                ppfn_kwargs = default_ppfn_kwargs)


