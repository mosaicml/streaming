### [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4)

1. Run the [write.py](https://github.com/mosaicml/streaming/blob/main/examples/text/c4/write.py) script as shown below. The script downloads the raw format with `train` and `val` splits from HuggingFace hub and converts to StreamingDataset MDS format into their own split directories. For more advanced use cases, please see the supported arguments for [c4.py](https://github.com/mosaicml/streaming/blob/main/examples/text/c4/convert/write.py) and modify as necessary.
    <!--pytest.mark.skip-->
    ```
    python write.py --out_root <local or remote directory path to save output MDS shard files>
    ```
