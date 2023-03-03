# Dataset preparation

To use Streaming Dataset we must first convert the dataset from its native format to MosaicML's Streaming Dataset format called Mosaic Dataset Shard (MDS). Once in MDS format, we can access the dataset from the local file system (disk network attached storage, etc.) or object store (GCS, OCS, S3, etc.).  From object store, data can be streamed to train deep learning models and it all just works efficiently.

Check out steps below for information on converting common NLP datasets to MDS format.  Please see [MDSWriter()](https://streaming.docs.mosaicml.com/en/latest/api_reference/generated/streaming.MDSWriter.html) parameters for details on advanced usage.

## NLP Dataset Conversion Examples

### [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4)

1. Run the [c4.py](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/c4.py) script as shown below. The script downloads the raw format with `train` and `val` splits from HuggingFace hub and converts to StreamingDataset MDS format into their own split directories. For more advanced use cases, please see the supported arguments for [c4.py](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/c4.py) and modify as necessary.
    <!--pytest.mark.skip-->
    ```
    python c4.py --out_root <local or remote directory path to save output MDS shard files>
    ```

### [Wikipedia](https://huggingface.co/datasets/wikipedia)

1. Download English Wikipedia 2020-01-01 from [here](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v).
2. Unzip the file `results_text.zip` as shown below.
    <!--pytest.mark.skip-->
    ```bash
    unzip results_text.zip
    ```

    Listing the output should show the following directory structure:
    <!--pytest.mark.skip-->
    ```bash
    ├── eval.txt
    ├── part-00000-of-00500
    ├── part-00001-of-00500
    ├── part-00002-of-00500
    ├── .....
    ├── part-00498-of-00500
    └── part-00499-of-00500
        ```

3. Run the [enwiki_text.py](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/enwiki_text.py) script. The script converts the `train` and `val` dataset splits into their own split directories. For more advanced use cases, please see the supported arguments for [enwiki_text.py](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/enwiki_text.py) and modify as necessary.
    <!--pytest.mark.skip-->
    ```
    python enwiki_text.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```

### [Pile](https://pile.eleuther.ai/)

1. Download the Pile dataset from [here](https://the-eye.eu/public/AI/pile/).

    Listing the output should show the following directory structure:
    <!--pytest.mark.skip-->
    ```bash
    ├── SHA256SUMS.txt
    ├── test.jsonl.zst
    ├── train
    │   ├── 00.jsonl.zst
    │   ├── 01.jsonl.zst
    │   ├── 02.jsonl.zst
    │   ├── 03.jsonl.zst
    │   ├── .....
    │   ├── 28.jsonl.zst
    │   └── 29.jsonl.zst
    └── val.jsonl.zst
    ```

2. Run the [pile.py](https://github.com/mosaicml/stireaming/blob/main/streaming/text/convert/pile.py) script. The script converts the `train`, `test`, and `val` dataset splits into their own split directories. For more advanced use cases, please see the supported arguments for [pile.py](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/pile.py) and modify as necessary.

    <!--pytest.mark.skip-->
    ```bash
    python pile.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
