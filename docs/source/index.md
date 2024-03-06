% Streaming documentation master file

# Streaming

StreamingDataset makes training on large datasets from cloud storage as fast, cheap, and scalable as possible. It’s specially designed for multi-node, distributed training of large models—maximizing correctness guarantees, performance, flexibility, and ease of use. Now, you can efficiently train anywhere, independent of where your dataset lives. Just train on the data you need, right when you need it.

StreamingDataset is compatible with any data type, including **images, text, video, and multimodal data**. With support for major cloud storage providers ([AWS](https://aws.amazon.com/s3/), [OCI](https://www.oracle.com/cloud/storage/object-storage/), [GCS](https://cloud.google.com/storage), [Azure](https://azure.microsoft.com/en-us/products/storage/blobs), and any S3 compatible object store such as [Cloudflare R2](https://www.cloudflare.com/products/r2/), [Coreweave](https://docs.coreweave.com/storage/object-storage), [Backblaze b2](https://www.backblaze.com/b2/cloud-storage.html), etc. ) and designed as a drop-in replacement for your PyTorch [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) class, StreamingDataset seamlessly integrates into your existing training workflows.

<!--pytest.mark.skip-->
```python
from torch.utils.data import DataLoader
from streaming import StreamingDataset

dataloader = DataLoader(dataset=StreamingDataset(remote='s3://...'))
```

## **💾** Installation
1. Set up your Python development environment.
2. Install Streaming with `pip`:
```
pip install mosaicml-streaming
```
3. Verify the installation with:
```
python -c "import streaming; print(streaming.__version__)"
```
4. Jump to our [Quick Start](getting_started/quick_start.md) and [Main Concepts](getting_started/main_concepts.md) guides.

## **🔑** Key Features

- **Elastic Determinism**: Samples are in the same order regardless of the number of GPUs, nodes, or CPU workers. This makes it simple to reproduce and debug training runs and loss spikes. You can load a checkpoint trained on 64 GPUs and debug on 8 GPUs with complete reproducibility.
- **Instant Mid-Epoch Resumption**: Resume training in seconds, not hours, in the middle of a long training run. Minimizing resumption latency saves thousands of dollars in egress fees and idle GPU compute time compared to existing solutions.
- **High throughput**: Our MDS format cuts extraneous work to the bone, resulting in ultra-low sample retrieval latency and higher throughput compared to alternatives.
- **Effective Shuffling**: Model convergence using StreamingDataset is just as good as using local disk, thanks to our [specialized shuffling algorithms](dataset_configuration/sampling_and_shuffling.md#Shuffling). StreamingDataset's shuffling reduces egress costs, preserves shuffle quality, and runs efficiently, whereas alternative solutions force tradeoffs between these factors.
- **Random access**: Access samples right when you need them -- simply call `dataset[i]` to get sample `i`. You can also fetch data on the fly by providing NumPy style indexing to `StreamingDataset`.
- **Flexible data mixing**: During streaming, different data sources are shuffled and mixed seamlessly just-in-time. Control how datasets are combined using our [batching](dataset_configuration/mixing_datasets.md/#Batching-methods) and [sampling](dataset_configuration/sampling_and_shuffling.md#Sampling-methods) methods.
- **Disk usage limits**: Dynamically delete least recently used shards in order to keep disk usage under a specified limit.

## Community

Streaming is part of the broader ML/AI community, and we welcome any contributions, pull requests, and issues.

If you have any questions, please feel free to reach out to us on [Twitter](https://twitter.com/mosaicml), 
[Email](mailto:community%40mosaicml.com), or [Slack](https://mosaicml.me/slack)!

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Overview

   getting_started/quick_start.md
   getting_started/main_concepts.md
   getting_started/faqs_and_tips.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Preparing Datasets

   preparing_datasets/dataset_format.md
   preparing_datasets/basic_dataset_conversion.md
   preparing_datasets/parallel_dataset_conversion.ipynb
   preparing_datasets/spark_dataframe_to_mds.ipynb

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Dataset Configuration

   dataset_configuration/shard_retrieval.md
   dataset_configuration/sampling_and_shuffling.md
   dataset_configuration/mixing_data_sources.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Distributed Training

   distributed_training/requirements.md
   distributed_training/with_launchers.md
   distributed_training/elastic_determinism.md
   distributed_training/fast_resumption.md
   distributed_training/performance_tuning.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: How-to Guides

   how_to_guides/configure_cloud_storage_credentials.md
   how_to_guides/llm_dataset_conversion.md
   how_to_guides/image_dataset_conversion.md

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
