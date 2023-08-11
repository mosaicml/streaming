# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import functools
import os.path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from pyspark.sql.types import StringType, StructField, StructType

from streaming.base.converters.dataframeToMDS import dataframeToMDS, default_mds_kwargs


def jsonToMDS(input_path: str,
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
              udf_iterable: Callable = None,
              udf_kwargs: Dict = None):

    import pyspark
    spark = pyspark.sql.SparkSession.builder.getOrCreate()

    dataframe = spark.read.schema(schema).json(input_path)

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
