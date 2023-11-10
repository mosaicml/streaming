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

3. Run the [write.py](https://github.com/mosaicml/streaming/blob/main/examples/text/enwiki_txt/write.py) script. The script converts the `train` and `val` dataset splits into their own split directories. For more advanced use cases, please see the supported arguments for [write.py](https://github.com/mosaicml/streaming/blob/main/examples/text/enwiki_txt/write.py) and modify as necessary.
    <!--pytest.mark.skip-->
    ```
    python write.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
