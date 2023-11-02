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

2. Run the [pile.py](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/pile.py) script. The script converts the `train`, `test`, and `val` dataset splits into their own split directories. For more advanced use cases, please see the supported arguments for [pile.py](https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/pile.py) and modify as necessary.

    <!--pytest.mark.skip-->
    ```bash
    python pile.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
