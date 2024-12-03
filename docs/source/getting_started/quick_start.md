# 🚀 Quick Start

Start training your model with Streaming in just a few steps!

1. Convert your raw dataset into one of our supported file formats. Here, we convert an image dataset to MDS (Mosaic Data Shard) format.
    <!--pytest.mark.skip-->
    ```python
    import numpy as np
    from PIL import Image
    from uuid import uuid4
    from streaming import MDSWriter

    # Local or remote directory path to store the output compressed files.
    out_root = 'dirname'

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

2. Replace the original {class}`torch.utils.data.IterableDataset` with your new {class}`streaming.StreamingDataset`. Point it to the dataset written out above, and specify the `batch_size` to StreamingDataset and the DataLoader.
    <!--pytest.mark.skip-->
    ```python
    from torch.utils.data import DataLoader
    from streaming import StreamingDataset

    # Remote directory where dataset is stored, from above
    remote_dir = 's3://path/to/dataset'

    # Local directory where dataset is cached during training
    local_dir = '/local/cache/path'
    dataset = StreamingDataset(local=local_dir, remote=remote_dir, batch_size=1, split=None, shuffle=True)

    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=1)
    ```

That's it! For additional details on using Streaming, check out the [Main Concepts](main_concepts.md) page and [How-to Guides](../how_to_guides/cifar10.ipynb).

We also have starter code for the following popular datasets, which can be found in the `streaming` [directory](https://github.com/mosaicml/streaming/tree/main/streaming):

| Dataset | Task | Read | Write |
| --- | --- | --- | --- |
| LAION-400M | Text and image | [Read](https://github.com/mosaicml/diffusion-benchmark/blob/main/data.py) | [Write](https://github.com/mosaicml/streaming/tree/main/streaming/multimodal/convert/laion/laion400m) |
| WebVid | Text and video | [Read](https://github.com/mosaicml/streaming/blob/main/streaming/multimodal/webvid.py) | [Write](https://github.com/mosaicml/streaming/blob/main/streaming/multimodal/convert/webvid.py) |
| C4 | Text | [Read](https://github.com/mosaicml/streaming/blob/main/streaming/text/c4.py) | [Write](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/c4.py) |
| EnWiki | Text | [Read](https://github.com/mosaicml/streaming/blob/main/streaming/text/enwiki.py) | [Write](https://github.com/mosaicml/streaming/tree/main/streaming/text/convert/enwiki) |
| Pile | Text | [Read](https://github.com/mosaicml/streaming/blob/main/streaming/text/pile.py) | [Write](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/pile.py)
| ADE20K | Image segmentation | [Read](https://github.com/mosaicml/streaming/blob/main/streaming/vision/ade20k.py) | [Write](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/ade20k.py)
| CIFAR10 | Image classification | [Read](https://github.com/mosaicml/streaming/blob/main/streaming/vision/cifar10.py) | [Write](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/cifar10.py) |
| COCO | Image classification | [Read](https://github.com/mosaicml/streaming/blob/main/streaming/vision/coco.py) | [Write](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/coco.py) |
| ImageNet | Image classification | [Read](https://github.com/mosaicml/streaming/blob/main/streaming/vision/imagenet.py) | [Write](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/imagenet.py) |

**To start training on these datasets:**

1. Convert raw data into .mds format using the corresponding script from the `convert` directory.

For example:

<!--pytest.mark.skip-->
```bash
$ python -m streaming.multimodal.convert.webvid --in <CSV file> --out <MDS output directory>
```

2. Import dataset class to start training the model.

<!--pytest.mark.skip-->
```python
from streaming.multimodal import StreamingInsideWebVid
dataset = StreamingInsideWebVid(local=local, remote=remote, batch_size=1, shuffle=True)
```

Happy training!
