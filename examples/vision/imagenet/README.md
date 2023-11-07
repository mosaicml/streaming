### [ImageNet](https://www.image-net.org/)

1. Download the ImageNet dataset from [here](https://image-net.org/download.php). Two files are needed, `ILSVRC2012_img_train.tar` for training and `ILSVRC2012_img_val.tar` for validation. Next untar both the files as shown below.
    <!--pytest.mark.skip-->
    ```bash
    mkdir val
    mv ILSVRC2012_img_val.tar val/
    tar -xvf ILSVRC2012_img_val.tar -C val/
    rm ILSVRC2012_img_val.tar

    mkdir train
    mv ILSVRC2012_img_train.tar train/
    tar -xvf ILSVRC2012_img_train.tar -C train/
    rm ILSVRC2012_img_train.tar
    ```

    Listing the output should show the following directory structure:
    <!--pytest.mark.skip-->
    ```bash
    ├── train/
      ├── n01440764
      │   ├── n01440764_10026.JPEG
      │   ├── n01440764_10027.JPEG
      │   ├── ......
      ├── ......
    ├── val/
      ├── n01440764
      │   ├── ILSVRC2012_val_00000293.JPEG
      │   ├── ILSVRC2012_val_00002138.JPEG
      │   ├── ......
      ├── ......
    ```

2. Run the [imagenet.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/imagenet.py) script as shown below. The script converts the `train` and `val` dataset splits into their own directories. For advanced uses cases, please see the supported arguments for [imagenet.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/imagenet.py) and modify as needed.
    <!--pytest.mark.skip-->
    ```
    python imagenet.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
