% Streaming documentation master file

# Streaming

StreamingDataset helps to make training on large datasets from cloud storage as fast, cheap, and scalable as possible. It’s specially designed for multi-node, distributed training for large models—maximizing correctness guarantees, performance, and ease of use. Now, you can efficiently train anywhere, independent of your training data location. Just stream in the data you need, when you need it.

StreamingDataset is compatible with any data type, including **images, text, video, and multimodal data**. With support for major cloud storage providers ([AWS](https://aws.amazon.com/s3/), [OCI](https://www.oracle.com/cloud/storage/object-storage/), [GCS](https://cloud.google.com/storage), [Azure](https://azure.microsoft.com/en-us/products/storage/blobs), and any S3 compatible object store such as [Cloudflare R2](https://www.cloudflare.com/products/r2/), [Coreweave](https://docs.coreweave.com/storage/object-storage), [Backblaze b2](https://www.backblaze.com/b2/cloud-storage.html), etc. ) and designed as a drop-in replacement for your PyTorch [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) class, StreamingDataset seamlessly integrates into your existing training workflows.

<!--pytest.mark.skip-->
```python
from torch.utils.data import DataLoader
from streaming import StreamingDataset

dataloader = DataLoader(dataset=StreamingDataset(remote='s3://...'))
```




## **🔑** Key Features

- **True Determinism**: Samples are in the same order regardless of the number of GPUs, nodes, or CPU workers. This makes it easier to reproduce and debug training runs and loss spikes and load a checkpoint trained on 64 GPUs and debug on 8 GPUs with reproducibility.
- **Instant Mid-Epoch Resumption**: Resume training in seconds, not hours, in the middle of a long training run. Minimizing resumption latency can save thousands of dollars in egress fees and idle GPU compute time compared to existing solutions.
- **High throughput**: Our MDS format cuts extraneous work to the bone, resulting in ultra-low sample latency and higher throughput compared to alternatives for workloads bottlenecked by the dataloader.
- **Equal Convergence**: Model convergence from using StreamingDataset is just as good as using local disk, thanks to our shuffling algorithm. StreamingDataset shuffles across all samples assigned to a node, whereas alternative solutions only shuffle samples in a smaller pool (within a single process).
- **Random access**: Access the data you need when you need it. Even if a sample isn’t downloaded yet, you can access `dataset[i]` to get sample `i`.
- **Numpy style indexing**: Fetch data on the fly by providing a NumPy style indexing to `StreamingDataset`.
- **Seamless data mixing**: During streaming, the different datasets are streamed, shuffled, and mixed seamlessly just-in-time.
- **Disk usage limits**: Dynamically delete least recently used shards in order to keep disk usage under a specified limit.

To get started, please checkout our [Quick Start](getting_started/quick_start.md) and [User Guide](getting_started/user_guide.md).

## Community

Streaming is part of the broader Machine Learning community, and we welcome any contributions, pull requests, and issues.

If you have any questions, please feel free to reach out to us on [Twitter](https://twitter.com/mosaicml), 
[Email](mailto:community%40mosaicml.com), or [Slack](https://mosaicml.me/slack)!

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation.md
   getting_started/quick_start.md
   getting_started/user_guide.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Fundamentals

   fundamentals/dataset_format.md
   fundamentals/dataset_conversion_guide.md
   fundamentals/compression.md
   fundamentals/hashing.md
   fundamentals/environments.md
   fundamentals/shuffling.md
   fundamentals/sampling.md
   fundamentals/batching.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: How-to Guides

   how_to_guides/configure_cloud_storage_credentials.md
   how_to_guides/dataset_conversion_to_mds_format.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/cifar10.ipynb
   examples/facesynthetics.ipynb
   examples/synthetic_nlp.ipynb
   examples/multiprocess_dataset_conversion.ipynb

.. toctree::
   :hidden:
   :caption: API Reference
   :maxdepth: 1
   :glob:

   api_reference/*

.. _Twitter: https://twitter.com/mosaicml
.. _Email: mailto:community@mosaicml.com
.. _Slack: https://mosaicml.me/slack
```
