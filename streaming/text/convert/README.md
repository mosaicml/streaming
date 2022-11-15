# Dataset preparation

To use Streaming Dataset, We first convert the dataset from its native format to MosaicML's streaming dataset format (a collection of binary `.mds` files). Once in `.mds` format, we can store the dataset in a central location (filesystem, S3, GCS, etc.) and stream the data to any compute cluster, with any number of devices, and any number of CPU workers, and it all ~ just works ~ .

## Converting C4 to streaming dataset .mds format

To make yourself a copy of C4, use `c4.py` as shown below with the minimalist arguments. For advanced users, you can look at the supported arguments for `c4.py` and change according to your own needs. The `c4.py` script uses the subset `en`.

```
# It downloads the 'raw format train' and 'val' split and convert to StreamingDataset format.
# For 'val' split, it will take anywhere from 10 sec to 1 min depending on your Internet bandwidth.
# For 'train' split, it will take anywhere from 1-to-many hours depending on bandwidth, # CPUs, etc.
# Once the script completes, you should see a dataset folder `./my-copy-c4/val` that is ~0.5GB and
# `./my-copy-c4/train` this is ~800GB.

python c4.py --out_root ./my-copy-c4
```
