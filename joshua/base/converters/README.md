### Spark Dataframe Conversion

Users can read datasets of any formats that Spark supports and convert the Spark dataframe to a Mosaic Streaming dataset. More specifically,

1. We enable converting a Spark DataFrame into an MDS format via the utility function [dataframe_to_mds](https://github.com/mosaicml/joshua/blob/main/joshua/base/converters/dataframe_to_mds.py). This utility function is flexible and supports a callable function, allowing modifications to the original data format. The function iterates over the callable, processes the modified data, and writes it in MDS format. For instance, it can be used with a tokenizer callable function that yields tokens as output.

2. Users are recommended to refer to the starting example [Jupyter notebook](https://github.com/mosaicml/joshua/blob/main/preparing_datasets/spark_dataframe_to_mds.ipynb) which demonstrates a complete workflow. It illustrates how to use Spark to read raw data into a Spark DataFrame and then convert it into the MDS format via the `dataframe_to_mds` function. In that tutorial, we also demonstrate the option to pass in a preprocessing tokenization job to the converter, which can be useful if materializing the intermediate dataframe is time consuming or taking extra development.
