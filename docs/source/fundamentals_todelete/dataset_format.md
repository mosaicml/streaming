# Dataset Format

## Introduction

To use StreamingDataset, one must convert raw data into one of our supported serialized dataset formats. With massive datasets, our serialization format choices are critical to the ultimate observed performance of the system. If we care about performance, we must own the format ourselves. In the Deep Learning model training, we need extremely low latency cold random access on individual samples' granularity to ensure the dataset is not a bottleneck.

StreamingDataset is compatible with any data type, including **images**, **text**, **video**, and **multimodal** data. StreamingDataset supports MDS (Mosaic Data Shard), CSV/TSV, and JSONL format, which can encode and decode most python objects.

## High-level design
During dataset conversion, StreamingDataset generates two types of files.
1. **index.json:** The index.json file contains metadata about the shard files, such as how many samples are in each shard file, the compression algorithm used, the shard filename, etc.
2. **One or more shard files:** Contains the data encoded into bytes. It is conceptualized as a table, each row being a sample. Columns have data types, which deserialize into any Python object such as int, str, PIL Image, etc. The shard file name starts with `shard.00000.<extension>` such as `shard.00000.mds`, `shard.00000.csv`, `shard.00000.jsonl`, etc., and number increments as it generates more shards.


## Formats
### 1. MDS
Mosaic Data Shard (MDS) is like a row-oriented parquet that reads a sample by reading its start/stop bytes from the header and then seeks to sample. The sample can be a singular data entity or a dictionary of key/value pairs where the key is a data field, and the value is data. Check out the [Dataset Conversion Guide](dataset_conversion_guide.md) section to understand more about MDSWriter. Most of the existing [dataset conversion](../how_to_guides/dataset_conversion_to_mds_format.md) script uses MDS format, which is highly flexible with any data type.

### 2. CSV/TSV
CSV (Comma-Separated Values) and TSV (Tab-Separated Values) are tabular data stored in plain-text form. CSV separates the data using delimiter `,` and TSV separates the data using delimiter `\t`.

### 3. JSONL
JSON Lines text format, also called newline-delimited JSON. JSON Lines consist of several lines where each line is a valid JSON object, separated by newline character `\n`. 
