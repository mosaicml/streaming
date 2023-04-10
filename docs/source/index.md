% Streaming documentation master file

# Streaming

StreamingDataset helps to make training on large datasets from cloud storage as fast, cheap, and scalable as possible. It’s specially designed for multi-node, distributed training for large models—maximizing correctness guarantees, performance, and ease of use. Now, you can efficiently train anywhere, independent of your training data location. Just stream in the data you need, when you need it.

StreamingDataset is compatible with any data type, including **images, text, video, and multimodal data**. With support for major cloud storage providers ([AWS](https://aws.amazon.com/s3/), [OCI](https://www.oracle.com/cloud/storage/object-storage/), and [GCS](https://cloud.google.com/storage) are supported today; [Azure](https://azure.microsoft.com/en-us/products/storage/blobs) is coming soon), and designed as a drop-in replacement for your PyTorch [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) class, StreamingDataset seamlessly integrates into your existing training workflows.

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

To get started, please checkout our [Quick Start](getting_started/quick_start.md) and [User Guide](getting_started/user_guide.md).

## Community

Streaming is part of the broader Machine Learning community, and we welcome any contributions, pull requests, and issues.

If you have any questions, please feel free to reach out to us on [Twitter](https://twitter.com/mosaicml), 
[Email](mailto:community%40mosaicml.com), or [Slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg)!

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
   :caption: Internal Guides

   internal_guides/environments.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: How-to Guides

   how_to_guides/configure_cloud_storage_cred.md
   how_to_guides/dataset_conversion_to_mds.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/cifar10.ipynb
   examples/facesynthetics.ipynb
   examples/synthetic_nlp.ipynb

.. toctree::
   :hidden:
   :caption: API Reference
   :maxdepth: 1
   :glob:

   api_reference/*

.. _Twitter: https://twitter.com/mosaicml
.. _Email: mailto:community@mosaicml.com
.. _Slack: https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg
```
