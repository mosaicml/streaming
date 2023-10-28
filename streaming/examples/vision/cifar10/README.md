### [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

1. Run the [cifar10.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/cifar10.py) script as shown below. The CIFAR10 dataset will be automatically downloaded if it doesn't exist locally. For advanced use cases, please see the supported arguments for [cifar10.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/cifar10.py) and modify as necessary.
    <!--pytest.mark.skip-->
    ```
    python cifar10.py --in_root <Above directory> --out_root <local or remote directory path to save output MDS shard files>
    ```
