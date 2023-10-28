### [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

1. Download the ADE20K dataset from [here](https://groups.csail.mit.edu/vision/datasets/ADE20K/).
2. Listing the output should show the following directory structure:
    <!--pytest.mark.skip-->
    ```bash
    ├── annotations
    │   ├── training
    │   └── validation
    └── images
        ├── training
        └── validation
    ```

3. Run the [ade20k.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/ade20k.py) script as shown below. The script converts the `train` and `val` dataset splits into their own directories. For advanced use cases, please see the supported arguments for [ade20k.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/ade20k.py) and modify according as necessary.
    <!--pytest.mark.skip-->
    ```
    python ade20k.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
