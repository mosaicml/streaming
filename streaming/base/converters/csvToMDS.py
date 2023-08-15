# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A utility to convert a csv dataset to MDS."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from streaming.base.converters.dataframeToMDS import dataframeToMDS


def csvToMDS(input_path: str,
             schema: StructType,
             out: Union[str, Tuple[str, str]],
             columns: Dict[str, str],
             partition_size: int = -1,
             merge_index: bool = True,
             sample_ratio: float = -1.0,
             keep_local: bool = False,
             compression: Optional[str] = None,
             hashes: Optional[List[str]] = None,
             size_limit: Optional[Union[int, str]] = 1 << 26,
             udf_iterable: Optional[Callable] = None,
             udf_kwargs: Dict[str, Any] = {}) -> None:
    """Execute a csv to MDS conversion process.

    Args:
        input_path (str): Path to source csv file.
        schema (StructType): User specified schema for spark.read.csv
        out (str | Tuple[str, str]): Output dataset directory to save shard files.
            1. If ``out`` is a local directory, shard files are saved locally.
            2. If ``out`` is a remote directory, a local temporary directory is created to
               cache the shard files and then the shard files are uploaded to a remote
               location. At the end, the temp directory is deleted once shards are uploaded.
            3. If ``out`` is a tuple of ``(local_dir, remote_dir)``, shard files are saved in the
               `local_dir` and also uploaded to a remote location.
        partition_size (int): The number of partitions to use during conversion. Default is -1, which does not do repartition.
        merge_index (bool): Whether to merge MDS index files. Default is True.
        sample_ratio (float): The fraction of data to randomly sample during conversion.
            Should be in the range (0, 1). Default is -1.0 (no sampling).
        columns (Dict[str, str]): Sample columns.
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

    Returns:
        None
    """
    spark = SparkSession.builder.getOrCreate()  # pyright: ignore

    dataframe = spark.read.schema(schema).csv(input_path)

    dataframeToMDS(dataframe,
                   partition_size=partition_size,
                   merge_index=merge_index,
                   sample_ratio=sample_ratio,
                   out=out,
                   columns=columns,
                   keep_local=keep_local,
                   compression=compression,
                   hashes=hashes,
                   size_limit=size_limit,
                   udf_iterable=udf_iterable,
                   udf_kwargs=udf_kwargs)
