# 🖼️ User Guide

At a very high level, one needs to convert a raw dataset into streaming format files and then use the same streaming format files using {class}`streaming.StreamingDataset` class for model training.

Streaming supports different dataset writers based on your need for conversion of raw datasets into a streaming format such as
- {class}`streaming.MDSWriter`: Writes the dataset into `.mds` (Mosaic Data Shard) extension. It supports various encoding/decoding formats(`str`, `int`, `bytes`, `jpeg`, `png`, `pil`, `pkl`, and `json`) which convert the data from that format to bytes and vice-versa.
- {class}`streaming.CSVWriter`: Writes the dataset into `.csv` (Comma Separated Values) extension. It supports various encoding/decoding formats(`str`, `int`, and `float`) which convert the data from that format to string and vice-versa.
- {class}`streaming.JSONWriter`: Writes the dataset into `.json` (JavaScript Object Notation) extension. It supports various encoding/decoding formats(`str`, `int`, and `float`).
- {class}`streaming.TSVWriter`: Writes the dataset into `.tsv` (Tab Separated Values) extension. It supports various encoding/decoding formats(`str`, `int`, and `float`) which convert the data from that format to string and vice-versa.
- {class}`streaming.XSVWriter`: Writes the dataset into `.xsv` (user defined Separated Values) extension. It supports various encoding/decoding formats(`str`, `int`, and `float`) which convert the data from that format to string and vice-versa.

For more information about writers and their parameters, look at the [API reference doc](../api_reference/streaming.rst).

After the dataset has been converted to one of our streaming formats, one just needs to instantiate the {class}`streaming.StreamingDataset` class by providing the dataset path of the streaming formats and use that dataset object in PyTorch {class}`torch.utils.data.DataLoader` class. For more information about `streaming.StreamingDataset` and its parameters, look at the {class}`streaming.StreamingDataset` API reference doc.

Streaming supports various dataset compression formats (Brotli, Bzip2, Gzip, Snappy, and Zstandard) that reduces downloading time and cloud egress fees. Additionally, Streaming also supports various hashing algorithms (SHA2, SHA3, MD5, xxHash, etc.) that ensures data integrity through cryptographic and non-cryptographic hashing algorithm.

Let's jump right into an example on how to convert a raw dataset into a streaming format and load the same streaming format dataset for model training.

## Writing a dataset to streaming format

This guide shows you how to use your custom StreamingDataset with {class}`streaming.MDSWriter`, but the steps would remain the same for other writers.

The {class}`streaming.MDSWriter` takes the raw dataset and converts it into a sharded `.mds` format for fast data access.

For this tutorial, let's create a Synthetic Classification dataset drawn from a normal distribution that returns a tuple of features and a label.

```python
import numpy as np

class RandomClassificationDataset:
    """Classification dataset drawn from a normal distribution.

    Args:
        shape: shape of features (default: (5, 1, 1))
        size: number of samples (default: 100)
        num_classes: number of classes (default: 2)
    """

    def __init__(self, shape=(1, 1, 1), size=100, num_classes=2):
        self.size = size
        self.x = np.random.randn(size, *shape)
        self.y = np.random.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
```

There are a few parameters that need to be initialized before {class}`streaming.MDSWriter` gets called. Some of the parameters are optional, and others are required parameters. Let's look at each of them where we start with two required parameters.

1. Provide the local filesystem directory path or a remote cloud provider storage path to store the compressed dataset files. If it is a remote path, the output files are automatically upload to a remote path.
    <!--pytest-codeblocks:cont-->
    ```python
    output_dir = 'test_output_dir'
    ```

2. Provide the column field as `Dict[str, str]`, which maps a feature name or label name with a streaming supported encoding type.
    <!--pytest-codeblocks:cont-->
    ```python
    columns = {'x': 'pkl', 'y': 'pkl'}
    ```

The below parameters are optional to {class}`streaming.MDSWriter`. Let's look at each one of them

1. Provide a name of a compression algorithm; the default is `None`. Streaming supports families of compression algorithms such as `br`, `gzip`, `snappy`, `zstd`, and `bz2` with the level of compression.
    <!--pytest-codeblocks:cont-->
    ```python
    compression = 'zstd:7'
    ```

2. Provide a name of a hashing algorithm; the default is `None`. Streaming supports families of hashing algorithm such as `sha`, `blake`, `md5`, `xxHash`, etc.
    <!--pytest-codeblocks:cont-->
    ```python
    hashes = ['sha1']
    ```

3. Provide a shard size limit, after which point to start a new shard.
    <!--pytest-codeblocks:cont-->
    ```python
    # Number act as a byte, e.g., 1024 bytes
    limit = 1024
    ```

Once the parameters are initialized, the last thing we need is a generator that iterates over the data sample.
<!--pytest-codeblocks:cont-->
```python
def each(samples):
    """Generator over each dataset sample.

    Args:
        samples (list): List of samples of (feature, label).

    Yields:
        Sample dicts.
    """
    for x, y in samples:
        yield {
            'x': x,
            'y': y,
        }
```

It's time to call the {class}`streaming.MDSWriter` with the above initialized parameters and write the samples by iterating over a dataset.
<!--pytest-codeblocks:cont-->
```python
from streaming.base import MDSWriter

dataset = RandomClassificationDataset()
with MDSWriter(out=output_dir, columns=columns, compression=compression, hashes=hashes, size_limit=limit) as out:
    for sample in each(dataset):
        out.write(sample)
```

Once the dataset has been written, the output directory contains two types of files. The first is an index.json file that contains the metadata of shards and second is the shard files. For example,
<!--pytest.mark.skip-->
```bash
dirname
├── index.json
├── shard.00000.mds.zstd
└── shard.00001.mds.zstd
```

## Loading a streaming dataset

After writing a dataset in the streaming format in the previous step and uploading to a cloud object storage as s3, we are ready to start loading the data.

To load the same dataset files that were created in the above steps, create a `CustomDataset` class by inheriting the {class}`streaming.StreamingDataset` class and override the `__getitem__(idx: int)` method to get the samples. The {class}`streaming.StreamingDataset` class requires two mandatory parameters which are `remote` which is a remote directory (S3 or local filesystem) where dataset is stored and `local` which is a local directory where dataset is cached during operation.
<!--pytest-codeblocks:cont-->
 ```python
from streaming import StreamingDataset

class CustomDataset(StreamingDataset):
    def __init__(self, local, remote):
        super().__init__(local, remote)

    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)
        return obj['x'], obj['y']
 ```

The next step is to Instantiate the `CustomDataset` class with local and remote paths.
<!--pytest-codeblocks:cont-->
```python
# Local filesystem directory where dataset is cached during operation
local = '/tmp/cache'

# Remote directory (S3 or local filesystem) where dataset is stored
remote='s3://mybucket/myfolder'

dataset = CustomDataset(local=local, remote=remote)
```

The final step is to pass the dataset to PyTorch {class}`torch.utils.data.DataLoader` and use this dataloader to train your model.
<!--pytest-codeblocks:cont-->
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset=dataset)
```

You've now seen an in-depth look at how to prepare and use streaming datasets with PyTorch. To continue learning about Streaming, please continue to explore our [examples](../examples/cifar10.ipynb/)!

## Other options

Please look at the API reference page for the complete list of {class}`streaming.StreamingDataset` supporting parameters.
