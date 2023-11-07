### [MS-COCO](https://cocodataset.org/#home)

1. Download the COCO 2017 dataset from [here](https://cocodataset.org/#download). Please download both the COCO images and annotations and unzip the files as shown below.
    <!--pytest.mark.skip-->
    ```bash
    mkdir coco
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    wget -c http://images.cocodataset.org/zips/train2017.zip
    wget -c http://images.cocodataset.org/zips/val2017.zip

    unzip annotations_trainval2017.zip
    unzip train2017.zip
    unzip val2017.zip

    rm annotations_trainval2017.zip
    rm train2017.zip
    rm val2017.zip
    ```

    Listing the output should show the following directory structure:
    <!--pytest.mark.skip-->
    ```bash
    ├── annotations
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017
    │   ├── 000000391895.jpg
    |   |── ...
    └── val2017
    │   ├── 000000000139.jpg
    |   |── ...
    ```

2. Run the [coco.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/coco.py) script as shown below. The script converts the `train` and `val` dataset splits into their own directories. For advanced use cases, please seet the supported arguments for [coco.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/coco.py) and modify as necessary.
    <!--pytest.mark.skip-->
    ```
    python coco.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
