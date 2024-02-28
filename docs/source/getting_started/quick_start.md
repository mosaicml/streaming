# 🚀 Quick Start

Start training your model with Streaming in just a few steps!

1. Convert your raw dataset into one of our supported file formats. Here, we convert an image dataset to MDS (Mosaic Data Shard) format.

    ```python
    import numpy as np
    from PIL import Image
    from shutil import rmtree
    from uuid import uuid4
    from streaming import MDSWriter

    # Local or remote directory path to store the output compressed files.
    # Here, we use a remote S3 path.
    out_root = 's3://path/to/dataset'

    # A dictionary of input fields to an Encoder/Decoder type
    columns = {
        'uuid': 'str',
        'img': 'jpeg',
        'clf': 'int'
    }

    # Compression algorithm name
    compression = 'zstd'

    # Generate random images and classes
    samples = [
        {
            'uuid': str(uuid4()),
            'img': Image.fromarray(np.random.randint(0, 256, (32, 48, 3), np.uint8)),
            'clf': np.random.randint(10),
        }
        for _ in range(1000)
    ]

    # Use `MDSWriter` to iterate through the input data and write to a collection of `.mds` files.
    with MDSWriter(out=out_root, columns=columns, compression=compression) as out:
        for sample in samples:
            out.write(sample)
    ```

2. Replace the original {class}`torch.utils.data.IterableDataset` with your new {class}`streaming.StreamingDataset`. Point it to the dataset written out above.
    <!--pytest.mark.skip-->
    ```python
    from torch.utils.data import DataLoader
    from streaming import StreamingDataset

    # Remote directory where dataset is stored, from above
    remote_dir = 's3://path/to/dataset'

    # Local directory where dataset is cached during training
    local_dir = '/local/cache/path'
    dataset = StreamingDataset(local=local_dir, remote=remote_dir, split=None, shuffle=True)

    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset)
    ```

That's it! For additional details on using Streaming, check out the [Main Concepts](main_concepts.md) page and [How-to Guides](../how_to_guides/llm_dataset_conversion.md).

Happy training!
