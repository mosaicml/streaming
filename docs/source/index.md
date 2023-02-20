% Streaming documentation master file

# Streaming

Welcome to MosaicML’s Streaming documentation page!  Streaming is a PyTorch compatible dataset that enables users to stream training data from
cloud-based object stores. Streaming can read files from local disk or from cloud-based object stores. As a drop-in replacement for your `Dataset` class, it's easy to get streaming:

<!--pytest.mark.skip-->
```python
dataloader = torch.utils.data.DataLoader(dataset=ImageStreamingDataset(remote='s3://...'))
```

For additional details, please see our [Quick Start](getting_started/quick_start.md) and [User Guide](getting_started/user_guide.md).

Streaming was originally developed as a part of MosaicML’s Composer training library and is a critical component of our efficient machine learning infrastructure.

## Installation

```bash
pip install mosaicml-streaming
```

## Key Benefits

- High performance, accurate streaming of training data from cloud storage
- Efficiently train anywhere, independent of training data location
- Cloud-native, no persistent storage required
- Enhanced data security—data exists ephemerally on training cluster

## Features

- Drop-in replacement for {class}`torch.utils.data.IterableDataset` class.
- Built-in support for popular open source datasets (e.g., ADE20K, C4, COCO, Enwiki, ImageNet, etc.).
- Support for various image, structured and unstructured text formats.
- Helper utilities to convert proprietary datasets to streaming format.
- Streaming dataset compression (e.g., gzip, snappy, zstd, bz2, etc.).
- Streaming dataset integrity (e.g., SHA2, SHA3, MD5, xxHash, etc.).

## Community

Streaming is part of the broader Machine Learning community, and we welcome any contributions, pull requests, and issues.

If you have any questions, please feel free to reach out to us on [Twitter](https://twitter.com/mosaicml), 
[Email](mailto:community%40mosaicml.com), or [Slack](https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg)!

```{eval-rst}
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/quick_start.md
   getting_started/user_guide.md

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: How-to Guides

   how_to_guides/cloud_providers.md

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
