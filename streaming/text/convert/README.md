# Dataset preparation

To use Streaming Dataset, We first convert the dataset from its native format to MosaicML's streaming dataset format (a collection of binary `.mds` files). Once in `.mds` format, we can store the dataset in a central location (filesystem, S3, GCS, etc.) and stream the data to any compute cluster, with any number of devices, and any number of CPU workers, and it all just works.

Follow the below steps to convert the Natural Language Processing dataset into a streaming MDS format. Also checkout the supported [MDSWriter()](https://streaming.docs.mosaicml.com/en/latest/api_reference/generated/streaming.MDSWriter.html) parameters for advanced usage.


## 1. [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4)

**Instructions:**

1. Run the [c4.py](c4.py) script as shown below with the minimalist arguments. The script downloads the raw format `train` and `val` split from HuggingFace hub and converts to StreamingDataset MDS format into their own split directory. For advanced users, you can look at the supported arguments for [c4.py](c4.py) and change according to your own needs.
    <!--pytest.mark.skip-->
    ```
    python c4.py --out_root <local or remote directory path to save output MDS shard files>
    ```

## 2. [Wikipedia](https://huggingface.co/datasets/wikipedia)

**Instructions:**

1. Download English Wikipedia 2020-01-01 from [here](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v).
2. unzip the file `results_text.zip` as shown below.
    <!--pytest.mark.skip-->
    ```bash
    unzip results_text.zip
    ```

    That will result in this directory structure:
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

3. Run the [enwiki_text.py](enwiki_text.py) script as shown below with the minimalist arguments. The script converts the `train` and `val` dataset split into their own split directory. For advanced users, you can look at the supported arguments for [enwiki_text.py](enwiki_text.py) and change according to your own needs.
    <!--pytest.mark.skip-->
    ```
    python enwiki_text.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```

## 3. [Pile](https://pile.eleuther.ai/)

**Instructions:**
1. Download the Pile dataset from [here](https://the-eye.eu/public/AI/pile/).

   That will result in this directory structure:
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

2. Run the [pile.py](pile.py) script as shown below with the minimalist arguments. The script converts the `train`, `test`, and `val` dataset split into their own split directory. For advanced users, you can look at the supported arguments for [pile.py](pile.py) and change according to your own needs.
    <!--pytest.mark.skip-->
    ```
    python pile.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
