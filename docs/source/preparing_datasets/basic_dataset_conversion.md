# Basic Dataset Conversion

This guide covers how to convert your raw data to MDS format using {class}`streaming.MDSWriter`. Writing to other supported shard formats is very similar. Read more about dataset shard formats in the [Dataset Format](dataset_format.md) guide. For a high-level explanation of how dataset writing works, check out the [main concepts](../getting_started/main_concepts.md#Dataset-conversion) page.

## Configuring dataset writing

Use {class}`streaming.MDSWriter` to convert raw data to MDS format. MDSWriter is like a native file writer; instead of writing the content line by line, MDSWriter writes the data sample by sample. It writes the data into shard files in a sequential manner (for example, `shard.00000.mds`, then `shard.00001.mds`, and so on). Configure {class}`streaming.MDSWriter` according to your requirements with the parameters below:

1. An `out` parameter is an output directory to save shard files. The `out` directory can be specified in three ways:
 * **Local path**: Shard files are stored locally.
 * **Remote path**: A local temporary directory is created to cache the shard files, and when shard creation is complete, they are uploaded to the remote location.
 * **`(local_dir, remote_dir)` tuple**: Shard files are saved in the specified `local_dir` and uploaded to `remote_dir`.

<!--pytest.mark.skip-->
```python
out = '/local/data'
out = 's3://bucket/data' # Will create a temporary local dir
out = ('/local/data', 'oci://bucket/data')
```

2. The optional `keep_local` parameter controls if you would like to keep the shard files locally after they have been uploaded to a remote cloud location. To save local disk space, this defaults to `False`.

3. A `column` parameter is a `dict` mapping a feature name or label name with a streaming supported encoding type. `MDSWriter` encodes your data to bytes, and at training time, data gets decoded back automatically to its original form. The `index.json` file stores `column` metadata for decoding. Supported encoding formats are:

| Category           | Name          | Class        | Notes                    |
|--------------------|---------------|--------------|--------------------------|
| Encoding           | 'bytes'       | `Bytes`      | no-op encoding           |
| Encoding           | 'str'         | `Str`        | stores in UTF-8          |
| Encoding           | 'int'         | `Int`        | Python `int`, uses `numpy.int64` for encoding       |
| Numpy Array        | 'ndarray:dtype:shape'     | `NDArray(dtype: Optional[str] = None, shape: Optional[tuple[int]] = None)`    | uses `numpy.ndarray`     |
| Numpy Unsigned Int | 'uint8'       | `UInt8`      | uses `numpy.uint8`       |
| Numpy Unsigned Int | 'uint16'      | `UInt16`     | uses `numpy.uint16`      |
| Numpy Unsigned Int | 'uint32'      | `Uint32`     | uses `numpy.uint32`      |
| Numpy Unsigned Int | 'uint64'      | `Uint64`     | uses  `numpy.uint64`     |
| Numpy Signed Int   | 'int8'        | `Int8`       | uses  `numpy.int8`       |
| Numpy Signed Int   | 'int16'       | `Int16`      | uses  `numpy.int16`      |
| Numpy Signed Int   | 'int32'       | `Int32`      | uses  `numpy.int32`      |
| Numpy Signed Int   | 'int64'       | `Int64`      | uses  `numpy.int64`      |
| Numpy Float        | 'float16'     | `Float16`    | uses  `numpy.float16`    |
| Numpy Float        | 'float32'     | `Float32`    | uses  `numpy.float32`    |
| Numpy Float        | 'float64'     | `Float64`    | uses  `numpy.float64`    |
| Numerical String   | 'str_int'     | `StrInt`     | stores in UTF-8          |
| Numerical String   | 'str_float'   | `StrFloat`   | stores in UTF-8          |
| Numerical String   | 'str_decimal' | `StrDecimal` | stores in UTF-8          |
| Image              | 'pil'         | `PIL`        | raw PIL image class ([link]((https://pillow.readthedocs.io/en/stable/reference/Image.html)))            |
| Image              | 'jpeg'        | `JPEG`       | PIL image as JPEG        |
| Image              | 'png'         | `PNG`        | PIL image as PNG         |
| Pickle             | 'pkl'         | `Pickle`     | arbitrary Python objects |
| JSON               | 'json'        | `JSON`       | arbitrary data as JSON   |

Here's an example where the field `x` is an image, and `y` is a class label, as an integer.
<!--pytest.mark.skip-->
```python
column = {
    'x': 'jpeg',
    'y': 'int8',
}
```

If the data type you need is not listed in the above table, then you can write your own data type class with `encode` and `decode` methods in it and patch it inside streaming. For example, let's say, you wanted to add a `complex128` data type (64 bits each for real and imaginary parts):

<!--pytest.mark.skip-->
```python
import numpy as np
from typing import Any

from streaming.base.format.mds.encodings import Encoding, _encodings

class Complex128(Encoding):

    def encode(self, obj: Any) -> bytes:
        return np.complex128(obj).tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.complex128)[0]

_encodings['complex128'] = Complex128
```

4. An optional shard `size_limit`, in bytes, for each *uncompressed* shard file. This defaults to 67 MB. Specify this as a number of bytes, either directly as an `int`, or a human-readable suffix:

<!--pytest.mark.skip-->
```python
size_limit = 1024 # 1kB limit, as an int
size_limit = '1kb' # 1kB limit, as a human-readable string
```
Shard file size depends on the dataset size, but generally, too small of a shard size creates a ton of shard files and heavy network overheads, and too large of a shard size creates fewer shard files, but downloads are less balanced. A shard size of between 50-100MB works well in practice.

5. An optional `compression` algorithm name (and level) if you would like to compress the shard files. This can reduce egress costs during training. StreamingDataset will uncompress shard files upon download during training. You can control whether to keep compressed shard files locally during training with the `keep_zip` flag -- more information [here](../dataset_configuration/shard_retrieval.md#Keeping-compressed-shards).

Supported compression algorithms:

| Name                                          | Code   | Min Level | Default Level | Max Level |
| --------------------------------------------- | ------ | --------- | ------------- | --------- |
| [Brotli](https://github.com/google/brotli)    | br     | 0         | 11            | 11        |
| [Bzip2](https://sourceware.org/bzip2/)        | bz2    | 1         | 9             | 9         |
| [Gzip](https://www.gzip.org/)                 | gz     | 0         | 9             | 9         |
| [Snappy](https://github.com/google/snappy)    | snappy | –         | –             | –         |
| [Zstandard](https://github.com/facebook/zstd) | zstd   | 1         | 3             | 22        |

The compression algorithm to use, if any, is specified by passing `code` or `code:level` as a string. For example:

<!--pytest.mark.skip-->
```python
compression = 'zstd' # zstd, defaults to level 3.
compression = 'zstd:9' # zstd, specifying level 9.
```
The higher the level, the higher the compression ratio. However, using higher compression levels will impact the compression speed. In our experience, `zstd` is optimal over the time-size Pareto frontier. Compression is most beneficial for text, whereas it is less helpful for other modalities like images.

6. An optional `hashes` list of algorithm names, used to verify data integrity. Hashes are saved in the `index.json` file. Hash verification during training is controlled with the `validate_hash` argument more information [here](../dataset_configuration/shard_retrieval.md#Hash-validation).

Available cryptographic hash functions:

| Hash       | Digest Bytes |
| ---------- | ------------ |
| 'blake2b'  | 64           |
| 'blake2s'  | 32           |
| 'md5'      | 16           |
| 'sha1'     | 20           |
| 'sha224'   | 28           |
| 'sha256'   | 32           |
| 'sha384'   | 48           |
| 'sha512'   | 64           |
| 'sha3_224' | 28           |
| 'sha3_256' | 32           |
| 'sha3_384' | 48           |
| 'sha3_512' | 64           |

Available non-cryptographic hash functions:

| Hash       | Digest Bytes |
| ---------- | ------------ |
| 'xxh32'    | 4            |
| 'xxh64'    | 8            |
| 'xxh128'   | 16           |
| 'xxh3_64'  | 8            |
| 'xxh3_128' | 16           |

As an example:

<!--pytest.mark.skip-->
```python
hashes = ['sha256', 'xxh64']
```

## Example: Writing a dataset to MDS format

Let's put it all together with an example. Here, we create a synthetic classification dataset that returns a tuple of features and a label.

```python
import numpy as np

class RandomClassificationDataset:
    """Classification dataset drawn from a normal distribution.

    Args:
        shape: data sample dimensions (default: (10,))
        size: number of samples (default: 10000)
        num_classes: number of classes (default: 2)
    """

    def __init__(self, shape=(10,), size=10000, num_classes=2):
        self.size = size
        self.x = np.random.randn(size, *shape)
        self.y = np.random.randint(0, num_classes, size)

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
```

Here, we write shards to a local directory. You can specify a remote path as well.
<!--pytest-codeblocks:cont-->
```python
output_dir = 'test_output_dir'
```

Specify the column encoding types for each sample and label:
<!--pytest-codeblocks:cont-->
```python
columns = {'x': 'pkl', 'y': 'int64'}
```

Optionally, specify a compression algorithm and level:
<!--pytest-codeblocks:cont-->
```python
compression = 'zstd:7' # compress shards with ZStandard, level 7
```

Optionally, specify a list of hash algorithms for verification:
<!--pytest-codeblocks:cont-->
```python
hashes = ['sha1'] # Use only SHA1 hashing on each shard
```

Optionally, provide a shard size limit, after which a new shard starts. In this small example, we use 10kb, but for production datasets 50-100MB is more appropriate.
<!--pytest-codeblocks:cont-->
```python
# Here we use a human-readable string, but we could also
# pass in an int specifying the number of bytes.
limit = '10kb'
```

It's time to call the {class}`streaming.MDSWriter` with the above initialized parameters and write the samples by iterating over a dataset.
<!--pytest-codeblocks:cont-->
```python
from streaming.base import MDSWriter

dataset = RandomClassificationDataset()
with MDSWriter(out=output_dir, columns=columns, compression=compression, hashes=hashes, size_limit=limit) as out:
    for x, y in dataset:
        out.write({'x': x, 'y': y})
```

Clean up after ourselves.
<!--pytest-codeblocks:cont-->
```
from shutil import rmtree

rmtree(output_dir)
```

Once the dataset has been written, the output directory contains an index.json file that contains shard metadata, the shard files themselves. For example,
<!--pytest.mark.skip-->
```bash
dirname
├── index.json
├── shard.00000.mds.zstd
└── shard.00001.mds.zstd
```

## Example: Writing `ndarray`s to MDS format

Here, we show how to write `ndarray`s to MDS format in three ways:
1. dynamic shape and dtype
2. dynamic shape but fixed dtype
3. fixed shape and dtype

Serializing ndarrays with fixed dtype and shape is more efficient than fixed dtype and dynamic shape, which is in turn more efficient than dynamic dtype and shape.

### Dynamic shape, dynamic dtype

The streaming encoding type, as the value in the `columns` dict, should simply be `ndarray`.

```python
import numpy as np
from streaming.base import MDSWriter, StreamingDataset
# Write to MDS
with MDSWriter(out='my_dataset1/',
               columns={'my_array': 'ndarray'}) as out:
    for i in range(42):
        # Dimension can change
        ndim = np.random.randint(1, 5)
        shape = np.random.randint(1, 5, ndim)
        shape = tuple(shape.tolist())
        my_array = np.random.normal(0, 1, shape)
        out.write({'my_array': my_array})

# Inspect dataset
dataset = StreamingDataset(local='my_dataset1/', batch_size=1)
for i in range(dataset.num_samples):
    print(dataset[i])
```

### Dynamic shape, fixed dtype

The streaming encoding type, as the value in the `columns` dict, should be `ndarray:dtype`. So in this example, it is `ndarray:int16`.

<!--pytest-codeblocks:cont-->
```python
# Write to MDS
with MDSWriter(out='my_dataset2/',
               columns={'my_array': 'ndarray:int16'}) as out:
    for i in range(42):
        # Dimension can change
        ndim = np.random.randint(1, 5)
        shape = np.random.randint(1, 5, ndim)
        shape = tuple(shape.tolist())
        # Datatype is fixed
        my_array = np.random.normal(0, 100, shape).astype(np.int16)
        out.write({'my_array': my_array})

# Inspect dataset
dataset = StreamingDataset(local='my_dataset2/', batch_size=1)
for i in range(dataset.num_samples):
    print(dataset[i])
```

### Fixed shape, fixed dtype

The streaming encoding type, as the value in the `columns` dict, should be `ndarray:dtype:shape`. So in this example, it is `ndarray:int16:3,3,3`.

<!--pytest-codeblocks:cont-->
```python
# Write to MDS
with MDSWriter(out='my_dataset3/',
               columns={'my_array': 'ndarray:int16:3,3,3'}) as out:
    for i in range(42):
        # Shape is fixed
        shape = 3, 3, 3
        # Datatype is fixed
        my_array = np.random.normal(0, 100, shape).astype(np.int16)
        out.write({'my_array': my_array})

# Inspect dataset
dataset = StreamingDataset(local='my_dataset3/', batch_size=1)
for i in range(dataset.num_samples):
    print(dataset[i])
```

We can see that the dataset is more efficiently serialized when we are more specific about array shape and datatype:

<!--pytest-codeblocks:cont-->
```python
import subprocess

# Dynamic shape, dynamic dtype uses most space
subprocess.run(['du', '-sh', 'my_dataset1'])

# Dynamic shape, fixed dtype uses less space
subprocess.run(['du', '-sh', 'my_dataset2'])

# Fixed shape, fixed dtype uses the least space
subprocess.run(['du', '-sh', 'my_dataset3'])
```

Clean up after ourselves.
<!--pytest-codeblocks:cont-->
```python
from shutil import rmtree

rmtree('my_dataset1')
rmtree('my_dataset2')
rmtree('my_dataset3')
```
