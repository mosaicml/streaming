# Dataset Conversion to MDS format

If you have not read the [dataset format guide](../fundamentals/dataset_format.md) and [introduction ot writer guide](../fundamentals/Introduction_to_writer.md), then we highly recommend you do so before you start.

To use Streaming Dataset we must first convert the dataset from its native format to MosaicML's Streaming Dataset format called Mosaic Dataset Shard (MDS). Once in MDS format, we can access the dataset from the local file system (disk network attached storage, etc.) or object store (GCS, OCS, S3, etc.).  From object store, data can be streamed to train deep learning models and it all just works.

## Convert a raw data into MDS format

Let's look at the steps one needs to perform to convert their raw data into an MDS format.

1. Get the raw dataset, either you can download all locally or create an iterator which downloads on the fly.
2. For the raw dataset, you need some form of iterator which fetches one sample at a time.
3. Convert the raw sample in the form of `column` field.
4. Instantiate MDSWriter and call the `write` method to write a raw sample one at a time.

Checkout the [user guide](../getting_started/user_guide.md) section which contains a simplistic example for the data conversion using single process. For multiprocess dataset conversion example, checkout [this](../examples/multiprocess_dataset_conversion.ipynb) tutorial.


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
