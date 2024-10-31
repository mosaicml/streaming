# Dataset Format

## Introduction

To use StreamingDataset, one must convert raw data into one of our supported serialized dataset formats. With massive datasets, our serialization format choices are critical to the ultimate observed performance of the system. For deep learning models, we need extremely low latency cold random access of individual samples granularity to ensure that dataloading is not a bottleneck to training.

StreamingDataset is compatible with any data type, including **images**, **text**, **video**, and **multimodal** data. StreamingDataset supports the following formats:
 * MDS (Mosaic Data Shard, most performant), through {class}`joshua.MDSWriter`
 * CSV/TSV, through {class}`joshua.CSVWriter` or {class}`joshua.TSVWriter`
 * JSONL, through {class}`joshua.JSONWriter`

These formats can encode and decode most Python objects.

For a high-level explanation of how dataset writing works, check out the [main concepts](../getting_started/main_concepts.md#Dataset-conversion) page. The [Dataset Conversion Guide](basic_dataset_conversion_guide.md) shows how to use the {class}`joshua.MDSWriter` to convert your raw data to supported file formats. For large datasets, use the [Parallel Dataset Conversion](parallel_dataset_conversion.ipynb) guide.


## Formats
### 1. MDS
Mosaic Data Shard (MDS) is our most performant file format for fast sample random-access, and stores data in serialized tabular form. A single sample is a dictionary of key/value pairs where the key is the column name, and the value is the sample's entry for that column. Use {class}`joshua.MDSWriter` for MDS.

### 2. CSV/TSV
CSV/TSV, or more generally XSV, is a plaintext tabular data format consisting of delimiter-separated values. For convenience, we have added two named sub-types which you will recognize as CSV (comma-delimited) and TSV (tab-delimited). To create datasets in these formats, use joshua.XSVWriter, joshua.CSVWriter, or joshua.TSVWriter.

### 3. JSONL
JSONL is a simple and popular dataset format in which each sample is a JSON dict terminated by a newline. Use {class}`joshua.JSONWriter` for JSONL.

## Metadata

Streaming also must store some metadata to keep track of a dataset's shards and samples. With MDS, only the `index.json` file is present, but with CSV/TSV and JSONL, additional files must also be stored which contain information about where specific samples are stored.

### The `index.json` file
As mentioned in the [main concepts](../getting_started/main_concepts.md#dataset-conversion) page, an `index.json` file is also created for each of shard files, containing information such as the number of shards, number of samples per shard, shard sizes, etc. An example `index.json` file, which has metadata for multiple MDS shards, and where samples contain only one column called "tokens" encoded as `Bytes`, is structured as below:
<!--pytest.mark.skip-->
```json
{
    "shards": [
        {   // Shard 0
            "column_encodings": ["bytes"],
            "column_names": ["tokens"],
            "column_sizes": [null],
            "compression": null,
            "format": "mds",
            "hashes": [],
            "raw_data": {
                "basename": "shard.00000.mds",
                "bytes": 67092637,
                "hashes": {}
            },
            "samples": 4093,
            "size_limit": 67108864,
            "version": 2,
            "zip_data": null
        },
        {   // Shard 1, very similar to Shard 0 metadata
            ...
            "raw_data": {
                "basename": "shard.00001.mds",
                "bytes": 67092637,
                "hashes": {}
            },
            ...
        },
    // and so on
    ]
}
```
