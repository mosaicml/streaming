# Dataset preparation

To use Streaming Dataset we must first convert the dataset from its native format to MosaicML's Streaming Dataset format called Mosaic Dataset Shard (MDS). Once in MDS format, we can access the dataset from the local file system (disk network attached storage, etc.) or object store (GCS, OCS, S3, etc.).  From object store, data can be streamed to train deep learning models and it all just works efficiently.

Check out steps below for information on converting common Computer Vision datasets to MDS format. Please see [MDSWriter()](https://streaming.docs.mosaicml.com/en/latest/api_reference/generated/streaming.MDSWriter.html) parameters for details on advanced usage.

## 1. [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

**Instructions:**

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

3. Run the [ade20k.py](ade20k.py) script as shown below with the minimalist arguments. The script converts the `train` and `val` dataset splits into their own directories. For advanced use cases, please see the supported arguments for [ade20k.py](ade20k.py) and modify according as necessary.
      <!--pytest.mark.skip-->
      ```
      python ade20k.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
      ```

## 2. [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

**Instructions:**

1. Run the [cifar10.py](cifar10.py) script as shown below with the minimalist arguments. The CIFAR10 dataset will be automatically downloaded if it doesn't exist locally. For advanced use cases, please see the supported arguments for [cifar10.py](cifar10.py) and modify as necessary.
    <!--pytest.mark.skip-->
    ```
    python cifar10.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```

## 3. [MS-COCO](https://cocodataset.org/#home)

**Instructions:**

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

2. Run the [coco.py](coco.py) script as shown below with the minimalist arguments. The script converts the `train` and `val` dataset splits into their own directories. For advanced use cases, please seet the supported arguments for [coco.py](coco.py) and modify as necessary.
    <!--pytest.mark.skip-->
    ```
    python coco.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```

## 4. [ImageNet](https://www.image-net.org/)

**Instructions:**

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

2. Run the [imagenet.py](imagenet.py) script as shown below with the minimalist arguments. The script converts the `train` and `val` dataset splits into their own directories. For advanced uses cases, please see the supported arguments for [imagenet.py](imagenet.py) and modify as needed.
    <!--pytest.mark.skip-->
    ```
    python imagenet.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
