<br />
<p align="center">
    <a href="https://github.com/mosaicml/streaming#gh-light-mode-only" class="only-light">
      <img src="https://storage.googleapis.com/docs.mosaicml.com/images/streaming-logo-light-mode.png" width="50%"/>
    </a>
    <!--pypi website does not support dark mode and does not understand GitHub tag. Hence, it renders both the images.
    The below tag is being used to remove the dark mode image on pypi website.-->
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
    <a href="https://github.com/mosaicml/streaming#gh-dark-mode-only" class="only-dark">
      <img src="https://storage.googleapis.com/docs.mosaicml.com/images/streaming-logo-dark-mode.png" width="50%"/>
    </a>
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->
</p>

<h2><p align="center">A Data Streaming Library for Efficient Neural Network Training</p></h2>

<h4><p align='center'>
<a href="https://www.mosaicml.com">[Website]</a>
- <a href="https://docs.mosaicml.com/projects/streaming/en/latest/getting_started/user_guide.html">[Getting Started]</a>
- <a href="https://docs.mosaicml.com/projects/streaming/">[Docs]
- <a href="https://www.mosaicml.com/team">[We're Hiring!]</a>
</p></h4>

<p align="center">
    <a href="https://pypi.org/project/mosaicml-streaming/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/mosaicml-streaming">
    </a>
    <a href="https://pypi.org/project/mosaicml-streaming/">
        <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/mosaicml-streaming">
    </a>
    <a href="https://github.com/mosaicml/streaming/actions?query=workflow%3ATest">
        <img alt="Unit test" src="https://github.com/mosaicml/streaming/actions/workflows/pytest.yaml/badge.svg">
    </a>
    <a href="https://pypi.org/project/mosaicml-streaming/">
        <img alt="PyPi Downloads" src="https://img.shields.io/pypi/dm/mosaicml-streaming">
    </a>
    <a href="https://docs.mosaicml.com/projects/streaming">
        <img alt="Documentation" src="https://readthedocs.org/projects/streaming/badge/?version=stable">
    </a>
    <a href="https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/mosaicml/streaming/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=slack">
    </a>
</p>
<br />

# 👋 Welcome
Streaming is a PyTorch compatible dataset that enables users to stream training data from cloud-based object stores. Streaming can read files from local disk or from cloud-based object stores. As a drop-in replacement for your PyTorch [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) class, it’s easy to get streaming:

<!--pytest.mark.skip-->
```python
dataloader = torch.utils.data.DataLoader(dataset=ImageStreamingDataset(remote='s3://...'))
```

Please check the [quick start guide](https://docs.mosaicml.com/projects/streaming/en/latest/getting_started/quick_start.html) and [user guide](https://docs.mosaicml.com/projects/streaming/en/latest/getting_started/user_guide.html) on how to use the Streaming Dataset.

# Key Benefits

- High performance, accurate streaming of training data from cloud storage
- Efficiently train anywhere, independent of training data location
- Cloud-native, no persistent storage required
- Enhanced data security—data exists ephemerally on training cluster


# 🚀 Quickstart

## 💾 Installation
Streaming is available with Pip:

<!--pytest.mark.skip-->
```bash
pip install mosaicml-streaming
```

# Examples
Please check our [Examples](https://docs.mosaicml.com/projects/streaming/) section for the end-to-end model training workflow using Streaming datasets.

# 📚 Documentation
Getting started guides, examples, API reference, and other useful information can be found in our [docs](https://docs.mosaicml.com/projects/streaming).

# 💫 Contributors
We welcome any contributions, pull requests, or issues!

To start contributing, see our [Contributing](https://github.com/mosaicml/streaming/blob/main/CONTRIBUTING.md) page.

P.S.: [We're hiring](https://mosaicml.com/jobs)!

# ✍️ Citation
```
@misc{mosaicml2022streaming,
    author = {The Mosaic ML Team},
    title = {streaming},
    year = {2022},
    howpublished = {\url{https://github.com/mosaicml/streaming/}},
}
```
