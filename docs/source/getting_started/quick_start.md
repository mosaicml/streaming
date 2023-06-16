# 🚀 Quick Start

Start training your model with the Streaming dataset in a few steps!

1. Convert your raw dataset into one of our supported streaming formats, for example, `mds` (Mosaic Data Shard) format.

    ```python
    import numpy as np
    from PIL import Image
    from shutil import rmtree
    from uuid import uuid4
    from streaming import MDSWriter

    # Local or remote directory path to store the output compressed files.
    # For remote directory, the output files are automatically upload to a remote cloud storage
    # location.
    out_root = 'dirname'

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
    with MDSWriter(out=out_root, columns=columns, compression=compression, hashes=hashes) as out:
        for sample in samples:
            out.write(sample)

    # Clean up
    rmtree(out_root)
    ```

2. Replace the original {class}`torch.utils.data.IterableDataset` with your new {class}`streaming.StreamingDataset`.
    <!--pytest.mark.skip-->
    ```python
    from torch.utils.data import DataLoader
    from streaming import StreamingDataset

    # Remote directory (S3 or local filesystem) where dataset is stored
    remote_dir = 's3://datapath'

    # Local directory where dataset is cached during operation
    local_dir = 'local_dir'
    dataset = StreamingDataset(local=local_dir, remote=remote_dir, split=None, shuffle=True)

    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset)
    ```

That's it! For additional details on using {mod}`streaming`, please check out our [User Guide](user_guide.md) and [Examples](../examples/cifar10.ipynb).

Happy training!
