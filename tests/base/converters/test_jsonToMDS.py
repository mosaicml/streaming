# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from tempfile import NamedTemporaryFile, mkdtemp

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from streaming.base.converters import jsonToMDS


class TestJsonToMDS:

    def test_end_to_end_conversion(self):
        spark = SparkSession.builder.appName('spark').getOrCreate()  # pyright: ignore
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

        df = spark.createDataFrame(data=data, schema=schema)

        temp_json_file = NamedTemporaryFile(delete=False)
        df.select(col('id'), col('dept')).write.mode('overwrite').json(temp_json_file.name)

        schema = StructType([
            StructField('id', StringType(), True),
            StructField('dept', StringType(), True),
        ])
        out = mkdtemp()
        jsonToMDS(temp_json_file.name,
                  schema=schema,
                  out=out,
                  columns={
                      'id': 'str',
                      'dept': 'str'
                  },
                  partition_size=2,
                  merge_index=True,
                  sample_ratio=-1.0,
                  keep_local=False,
                  compression='zstd:7',
                  hashes=['sha1', 'xxh64'],
                  size_limit=1 << 26)

        assert (os.path.exists(os.path.join(out, 'index.json'))), 'No merged index found'
        assert (len(os.listdir(out)) > 0), f'{out} is empty'
        for d in os.listdir(out):
            if os.path.isdir(os.path.join(out, d)):
                assert (os.path.exists(os.path.join(
                    out, d, 'index.json'))), f'No index.json found in subdirectory {d}'
