# 🚀 Quick Start

Here's how to get started!

There are mainly three steps to use the streaming dataset. First, you need to convert your raw dataset into one of the format we support, for example, `mds` format.

```python
import numpy as np
from PIL import Image
from uuid import uuid4
from streaming import MDSWriter

# Directory path to store the output compressed files
dirname = 'dirname'

# A dictionary of input fields to an Encoder/Decoder type
columns = {
    'uuid': 'str',
    'img': 'jpeg',
    'clf': 'int'
}

# Compression algorithm name
compression = 'zstd'

# Hash algorithm name
hashes = 'sha1', 'xxh64'

# Generates random images and classes for input sample
samples = [
    {
        'uuid': str(uuid4()),
        'img': Image.fromarray(np.random.randint(0, 256, (32, 48, 3), np.uint8)),
        'clf': np.random.randint(10),
    }
    for _ in range(1000)
]

# Call `MDSWriter` to iterate through the input data and write into a shard `mds` file
with MDSWriter(dirname, columns, compression, hashes) as out:
    for sample in samples:
        out.write(sample)
```

Second, upload the files or a folder to a cloud blob storage such as AWS S3.

Third, use the streaming Dataset within your PyTorch Dataloader during training time

```python
from torch.utils.data import DataLoader
from streaming import Dataset

# Remote directory (S3 or local filesystem) where dataset is stored
remote_dir = 's3://datapath'

# Local directory where dataset is cached during operation
local_dir = 'local_dir'
dataset = Dataset(local=local_dir, remote=remote_dir, split=None, shuffle=True)

# Create PyTorch DataLoader
dataloader = DataLoader(dataset)
```

See here for a more detailed guide to streaming dataset.
