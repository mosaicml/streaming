# User Guide

At a high level, one needs to create a dataset files compatible with a streaming and then load the same dataset files using {class}`streaming.Dataset` class.

Streaming supports different dataset writer based on your need such as {class}`streaming.MDSWriter`, {class}`streaming.CSVWriter`, {class}`streaming.JSONWriter`, {class}`streaming.TSVWriter`, and {class}`streaming.XSVWriter`. The {class}`streaming.MDSWriter` write the dataset into `.mds` extension, the {class}`streaming.CSVWriter` write the content in `.csv` format and so on. For the more information about writer and its parameters, look at the API reference doc.

For loading the dataset during model training, one needs to instantiate the {class}`streaming.Dataset` class with a dataset file path and provide that object to `dataset` parameter in PyTorch {class}`torch.utils.data.DataLoader` class. For the more information about dataset and its parameters, look at the API reference doc.

## Writing a dataset to streaming format

This guide shows you how to use your custom Dataset with {class}`streaming.MDSWriter`, but the steps would also remain the same for another writer.

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

1. Provide the Local filesystem directory path to store the compressed dataset
    <!--pytest-codeblocks:cont-->
    ```python
    output_dir = 'test_output_dir'
    ```

2. Provide the column field as `Dict[str, str]`, which maps a feature or label name with encoding type. All the supported encoding and decoding types can be found here.
    <!--pytest-codeblocks:cont-->
    ```python
    fields = {'x': 'pkl', 'y': 'pkl'}
    ```

The below parameters are optional to {class}`streaming.MDSWriter`. Let's look at each one of them

1. Provide a name of a compression algorithm; the default is `None`. We support families of compression algorithms, and all of them can be found here.
    <!--pytest-codeblocks:cont-->
    ```python
    compression = 'zstd:7'
    ```

2. Provide a name of a hashing algorithm; the default is `None`. All the supported compression algorithms can be found here.
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

It's time to call the {class}`streaming.MDSWriter` with the above initalized parameters and write the samples by iterating over a dataset.
<!--pytest-codeblocks:cont-->
```python
from streaming.base import MDSWriter

dataset = RandomClassificationDataset()
with MDSWriter(output_dir, fields, compression, hashes, limit) as out:
    for sample in each(dataset):
        out.write(sample)
```

Once the dataset has been written, the output directory contains two types of files. First are sharded files, and second is an index.json file that contains the metadata of shards. For example,
<!--pytest.mark.skip-->
```bash
dirname
├── index.json
├── shard.00000.mds.zstd
└── shard.00001.mds.zstd
```

Finally, upload the output directory to a cloud blob storage such as AWS S3. Below is one example of uploading a directory to an S3 bucket using [AWS CLI](https://aws.amazon.com/cli/).
<!--pytest.mark.skip-->
```bash
$ aws s3 cp dirname s3://mybucket/myfolder --recursive
```

## Loading a streaming dataset

After writing a dataset in the streaming format in the previous step and uploaded to a cloud object storage as s3, we are ready to start loading data.

To load the dataset files that we have written, we need to inherit the {class}`streaming.Dataset` class by creating a `CustomDataset` class and overriding the `__getitem__(idx: int)` method to get the sharded dataset. The {class}`streaming.Dataset` class requires to specify `local` and `remote` instance variables.
<!--pytest-codeblocks:cont-->
 ```python
from streaming.base import Dataset

class CustomDataset(Dataset):
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

## Other options

Please look at the API reference page for the complete list of {class}`streaming.Dataset` supporting parameters.
