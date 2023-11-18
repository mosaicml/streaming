# LAION-400M
[LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) is a dataset with CLIP-filtered 400 million image-text pairs. The LAION-400M dataset is entirely openly, freely accessible.
## Dataset preparation

To use Streaming Dataset, we must first convert the dataset from its native format to MosaicML's Streaming Dataset format called Mosaic Dataset Shard (MDS). Once in MDS format, we can access the dataset from the local file system (disk network attached storage, etc.) or object store (GCS, OCS, S3, etc.). From the object store, data can be streamed to train deep learning models, and it all works efficiently.

Check out the steps below for information on converting WebVid datasets to MDS format—also, checkout [MDSWriter](https://streaming.docs.mosaicml.com/en/latest/api_reference/generated/streaming.MDSWriter.html) parameters for details on advanced usage.

**1. Install dependencies**
Install package `img2dataset`.
<!--pytest.mark.skip-->
```
# Used for crawling.
pip3 install img2dataset==1.41.0

# Optional performance monitoring.
apt install bwm-ng htop iotop
```

**2. Get the streaming code**
<!--pytest.mark.skip-->
```
git clone https://github.com/mosaicml/streaming/
cd streaming/
```

**3. Download metadata from the-eye.eu (parquet format)**
<!--pytest.mark.skip-->
```
./examples/multimodal/laion400m/download_meta.sh
```

**4. Download data from the web (into parquet format, converting to mds format)**

The img2dataset download script saves samples in parquet files.
<!--pytest.mark.skip-->
```
./examples/multimodal/laion400m/download_data.sh
```

At the same time, do our conversion and uploading which uses MDS (you will want to run them at the same time, or disk usage can get excessive):
<!--pytest.mark.skip-->
```
./examples/multimodal/laion400m/convert_and_upload.sh
```

**Optional**
For system monitoring, run the below commands:

- **Monitor network i/o:** `bwm-ng`
- **Monitor CPU usage:** `htop`
- **Monitor disk i/o:** `iotop`
- **Monitor disk usage:** `df -h`
