# Dataset preparation

To use Streaming Dataset, We first convert the dataset from its native format to MosaicML's streaming dataset format (a collection of binary `.mds` files). Once in `.mds` format, we can store the dataset in a central location (filesystem, S3, GCS, or OCI) and stream the data to any compute cluster, with any number of devices, and any number of CPU workers, and it all just works.

Follow the below steps to convert the Computer Vision dataset into a streaming MDS format. Also checkout the supported [MDSWriter()](https://streaming.docs.mosaicml.com/en/latest/api_reference/generated/streaming.MDSWriter.html) parameters for advanced usage.

## 1. [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

**Instructions:**

1. Download the ADE20K dataset from [here](https://groups.csail.mit.edu/vision/datasets/ADE20K/).
2. That will result in this directory structure:

    ```bash
    ├── annotations
    │   ├── training
    │   └── validation
    └── images
        ├── training
        └── validation
    ```

3. Run the [ade20k.py](ade20k.py) script as shown below with the minimalist arguments. The script converts the `train` and `val` dataset split into their own directory. For advanced users, you can look at the supported arguments for [ade20k.py](ade20k.py) and change according to your own needs.

      ```
      python ade20k.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
      ```

## 2. [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

**Instructions:**

1. Run the [cifar10.py](cifar10.py) script as shown below with the minimalist arguments which also downloads the raw CIFAR10 dataset if it doesn't exist. For advanced users, you can look at the supported arguments for [cifar10.py](cifar10.py) and change according to your own needs.

    ```
    python cifar10.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```

## 3. [MS-COCO](https://cocodataset.org/#home)

**Instructions:**

1. Download the COCO 2017 dataset from [here](https://cocodataset.org/#download). Please download both the COCO images and annotations and unzip the files as shown below.

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

    That will result in this directory structure:

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

2. Run the [coco.py](coco.py) script as shown below with the minimalist arguments. The script converts the `train` and `val` dataset split into their own directory. For advanced users, you can look at the supported arguments for [coco.py](coco.py) and change according to your own needs.

    ```
    python coco.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```

## 4. [ImageNet](https://www.image-net.org/)

**Instructions:**

1. Download the ImageNet dataset from [here](https://image-net.org/download.php). More specifically, 2 files would be needed, `ILSVRC2012_img_train.tar` for training and `ILSVRC2012_img_val.tar` for validation. Then, untar both the files as shown below.

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

    That will result in this directory structure:

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

2. Run the [imagenet.py](imagenet.py) script as shown below with the minimalist arguments. The script converts the `train` and `val` dataset split into their own directory. For advanced users, you can look at the supported arguments for [imagenet.py](imagenet.py) and change according to your own needs.

    ```
    python imagenet.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
