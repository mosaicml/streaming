# Dataset Conversion to MDS format

To use Streaming Dataset we must first convert the dataset from its native format to MosaicML's Streaming Dataset format called Mosaic Dataset Shard (MDS). Once in MDS format, we can access the dataset from the local file system (disk network attached storage, etc.) or object store (GCS, OCS, S3, etc.).  From object store, data can be streamed to train deep learning models and it all just works.

We've already created conversion scripts that can be used to convert popular public datasets to MDS format.  Please see below for usage instructions.

## NLP Dataset Conversion Examples

```{include} ../../../streaming/text/convert/README.md
:start-line: 8
```

## Vision Dataset Conversion Examples

```{include} ../../../streaming/vision/convert/README.md
:start-line: 8
```

## Multimodal Dataset Conversion Examples
### [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/)
```{include} ../../../streaming/multimodal/convert/laion/laion400m/README.md
:start-line: 8
```
### [WebVid](https://m-bain.github.io/webvid-dataset/)
```{include} ../../../streaming/multimodal/convert/webvid/README.md
:start-line: 12
```
