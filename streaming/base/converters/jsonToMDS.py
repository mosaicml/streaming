# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A utility to convert a json dataset to MDS."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from streaming.base.converters.dataframeToMDS import dataframeToMDS


def jsonToMDS(input_path: str,
              schema: StructType,
              merge_index: bool = True,
              sample_ratio: float = -1.0,
              mds_kwargs: Dict[str, Any] = {},
              udf_iterable: Optional[Callable] = None,
              udf_kwargs: Dict[str, Any] = {}):
    """Execute a json to MDS conversion process.

    Args:
        input_path (str): Path to source csv file.
        schema (StructType): User specified schema for spark.read.csv
        merge_index (bool): Whether to merge MDS index files. Default is True.
        sample_ratio (float): The fraction of data to randomly sample during conversion.
            Should be in the range (0, 1). Default is -1.0 (no sampling).
        mds_kwargs (Dict): same arguments that would be passed to mdswriter
        udf_iterable (Callable or None): A user-defined function that returns an iterable over the dataframe. ppfn_kwargs is the k-v args for the method. Default is None.
        udf_kwargs (Dict): Additional keyword arguments to pass to the pandas processing
            function if provided. Default is an empty dictionary.

    Returns:
        None
    """
    spark = SparkSession.builder.getOrCreate()  # pyright: ignore

    dataframe = spark.read.schema(schema).json(input_path)

    dataframeToMDS(dataframe,
                   merge_index=merge_index,
                   sample_ratio=sample_ratio,
                   mds_kwargs = mds_kwargs,
                   udf_iterable=udf_iterable,
                   udf_kwargs=udf_kwargs)

