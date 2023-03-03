# Dataset Conversion to MDS format

To use Streaming Dataset we must first convert the dataset from its native format to MosaicML's Streaming Dataset format called Mosaic Dataset Shard (MDS). Once in MDS format, we can access the dataset from the local file system (disk network attached storage, etc.) or object store (GCS, OCS, S3, etc.).  From object store, data can be streamed to train deep learning models and it all just works efficiently.

Streaming repository contains ready to use scripts for different modality. Check out the below documents on the steps required to convert the dataset into an MDS format.

- [Computer Vision](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/README.md)
- [Natural Language Processing](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/README.md)
