### [Spark Dataframe](https://github.com/mosaicml/streaming/blob/main/examples/spark_dataframe_to_MDS.ipynb)

1. We enable converting a Spark DataFrame into an MDS format via a utility function `dataframeToMDS`. This utility function is flexible and supports a callable function, allowing modifications to the original data format. The function iterates over the callable, processes the modified data, and writes it in MDS format. For instance, it can be used with a tokenizer callable function that yields tokens as output.

Users are recommended to refer to the starting example Jupyter notebook which demonstrates a complete workflow. It illustrates how to use Spark to read raw data into a Spark DataFrame and then convert it into the MDS format via the `dataframeToMDS` function.
