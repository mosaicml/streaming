from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StructType, StructField

# Initialize Spark session
spark = SparkSession.builder.appName("NestedArrayDataFrame").getOrCreate()

# Define the schema with nested array types
schema = StructType([
    StructField("nested_float_array_1", ArrayType(ArrayType(FloatType())), True),
    StructField("nested_float_array_2", ArrayType(ArrayType(FloatType())), True),
    StructField("integer_array", ArrayType(IntegerType()), True)
])

# Create sample data
data = [
    (
        [[1.1, 2.2], [3.3, 4.4]],
        [[5.5, 6.6], [7.7, 8.8]],
        [1, 2, 3]
    ),
    (
        [[9.9, 10.1], [11.2, 12.3]],
        [[13.4, 14.5], [15.6, 16.7]],
        [4, 5, 6]
    )
]

# Create DataFrame with the sample data and schema
df = spark.createDataFrame(data, schema)

# Show the DataFrame
df.show(truncate=False)

# Print the schema
df.printSchema()

# Stop the Spark session
# spark.stop()

import numpy as np

mds_kwargs = {
    'out': '/tmp/jsontest',
    'keep_local': True,
    'columns': {'array_1': 'ndarray:float32',
                'array_2': 'ndarray:float32',
                'integer_array': 'ndarray:int32'
               }
}

def udf_iterable(df):
    records = df.to_dict('records')
    for sample in records:
        v = sample['nested_float_array_1']
        array1_float32 = np.vstack(v).astype(np.float32)

        v = sample['nested_float_array_2']
        array2_float32 = np.vstack(v).astype(np.float32)

        yield {'array_1': array1_float32, 'array_2': array2_float32, 'integer_array': sample['integer_array']}

from streaming.base.converters import dataframe_to_mds
_ = dataframe_to_mds(df.repartition(4),
                     merge_index=True,
                     mds_kwargs=mds_kwargs,
                     udf_iterable = udf_iterable)

