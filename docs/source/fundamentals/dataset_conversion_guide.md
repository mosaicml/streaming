# Dataset Conversion Guide

If you haven't read the [Dataset Format](dataset_format.md) guide, then we highly recommend doing so before you read this. 

## MDSWriter

To convert the dataset into MDS format, one must use {class}`streaming.MDSWriter`. MDSWriter is like a native file writer; instead of writing the content line by line, MDSWriter writes the data sample by sample. It writes the data into a first shard file (for example, `shard.00000.mds`), and once the shard file reaches a size limit, it creates a new shard file with a number incremented (for example, `shard.00001.mds`), and so on. {class}`streaming.MDSWriter` support various parameters you can tweak based on your requirements. Let's understand each parameter one by one:

1. An `out` parameter is an output dataset directory to save shard files. If the parameter is a local directory path, the shard files are stored locally. If the parameter is a remote directory, a local temporary directory is created to cache the shard files, and then the shard files are uploaded to a remote location. In the end, the temp directory is deleted once shards are uploaded. If the parameter is a tuple of `(local_dir, remote_dir)`, shard files are saved in the `local_dir` and uploaded to a remote location. As shard files are ready, it gets uploaded in the background to a remote location if provided. The user does not have to worry about uploading the shard files manually. `MDSWriter` also support a `keep_local` parameter where after uploading of an individual shard file is completed, you have the flexibility of deleting the shard file locally by providing `keep_local` to `False` (Default is `False`) to avoid running out of disk space.Checkout the [out](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html) parameter for more detail. For example, one can provide the `out` parameter as shown below:
<!--pytest.mark.skip-->
```python
out = '/tmp/data'
out = 's3://bucket/data'
out = {'/local/data', 'oci://bucket/data'}
```

2. A `column` parameter that maps a feature name or label name with a streaming supported encoding type. `MDSWriter` encodes your data from provided encoding type to bytes, and later it gets decoded back automatically to its original data type when calling `StreamingDataset`. The `index.json` file saves `column` information for decoding. Below is the list of supported encoding formats.

| Name   | Class  | Name    | Class   | Name | Class  |
| ------ | ------ | ------- | ------- | ---- | ------ |
| bytes  | `Bytes`  | int8    | `Int8`    | pil  | `PIL`    |
| str    | `Str`    | int16   | `Int16`   | jpeg | `JPEG`   |
| int    | `Int`    | int32   | `Int32`   | png  | `PNG`    |
| uint8  | `UInt8`  | int64   | `Int64`   | pkl  | `Pickle` |
| uint16 | `UInt16` | float16 | `Float16` | json | `JSON`   |
| uint32 | `UInt32` | float32 | `Float32` |      |        |
| uint64 | `UInt64` | float64 | `Float64` |      |        |

Below is one example where the feature name `x` is an image, and the label `y` is a class value.
<!--pytest.mark.skip-->
```python
column = {
    'x': 'jpeg',
    'y': 'int8'
}
```

**Advanced use-case:** If the data type you are interested in is not listed in the above table, then you can write your own data type class with `encode` and `decode` method in it and patch it inside streaming. For example, let say, you would like to write the same for `int32` data type.

<!--pytest.mark.skip-->
```python
import numpy as np
from typing import Any

from streaming.base.format.mds.encodings import Encoding, _encodings

class Int32(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.int32)

_encodings['int32'] = Int32
```


3. A `compression` algorithm name if you would like to compress the shard files. Check out the [compression](compression.md) document for more details.

4. A `hashes` algorithm name to verify data integrity. Check out the [hashing](hashing.md) document for additional details.

5. A shard `size_limit` in bytes for each shard file, after which point to start a new shard. Shard file size depends on the dataset size, but generally, too small of a shard size creates a ton of shard files and heavy network overheads, and too large of a shard size creates fewer shard files, but the training start time would increase since it has to wait for a shard file to get downloaded locally. Based on our intuition, the shard file size of 64Mb, and 128Mb play a balanced role.

6. A `keep_local` parameter if you would like to keep the shard files locally after it has been uploaded to a remote cloud location by MDSWriter.

This gives you a good understanding of {class}`streaming.MDSWriter` parameters. If you would like to convert your raw data into an MDS format, check out the [Dataset Conversion to MDS Format](../how_to_guides/dataset_conversion_to_mds_format.md) guide.
