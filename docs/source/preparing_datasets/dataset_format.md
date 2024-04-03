# Dataset Format

## Introduction

To use StreamingDataset, one must convert raw data into one of our supported serialized dataset formats. With massive datasets, our serialization format choices are critical to the ultimate observed performance of the system. For deep learning models, we need extremely low latency cold random access of individual samples granularity to ensure that dataloading is not a bottleneck to training.

StreamingDataset is compatible with any data type, including **images**, **text**, **video**, and **multimodal** data. StreamingDataset supports the following formats:
 * MDS (Mosaic Data Shard, most performant), through {class}`streaming.MDSWriter`
 * CSV/TSV, through {class}`streaming.CSVWriter` or {class}`streaming.TSVWriter`
 * JSONL, through {class}`streaming.JSONWriter`

These formats can encode and decode most Python objects.

For a high-level explanation of how dataset writing works, check out the [main concepts](../getting_started/main_concepts.md#Dataset-conversion) page. The [Dataset Conversion Guide](basic_dataset_conversion_guide.md) shows how to use the {class}`streaming.MDSWriter` to convert your raw data to supported file formats. For large datasets, use the [Parallel Dataset Conversion](parallel_dataset_conversion.ipynb) guide.


## Formats
### 1. MDS
Mosaic Data Shard (MDS) is our most performant file format for fast sample random-access. The sample can be a singular data entity or a dictionary of key/value pairs where the key is a data field, and the value is data. Use {class}`streaming.MDSWriter` for MDS.

### 2. CSV/TSV
CSV/TSV, or more generally XSV, is tabular data stored in plain-text form. Typically, CSV (Comma-Separated Values) and TSV (Tab-Separated Values) are used. CSV separates the data using delimiter `,` and TSV separates the data using delimiter `\t`. Use {class}`streaming.XSVWriter` for XSV, {class}`streaming.CSVWriter` for CSV, and {class}`streaming.TSVWriter` for TSV.

### 3. JSONL
JSON Lines text format, also called newline-delimited JSON, consists of several lines where each line is a valid JSON object, separated by newline character `\n`. Use {class}`streaming.JSONWriter` for JSONL.
