## Instructions:

### 1. Setup.

```
pip3 install img2dataset==1.41.0

apt install bwm-ng htop iotop
```

### 2. Download metadata from the-eye.eu (parquet format).

```
./streaming/text/convert/laion/laion400m/download_meta.sh
```

### 3. Download data from the web (into parquet format, converting to mds format).

The img2dataset download script saves samples in parquet files.

```
./streaming/text/convert/laion/laion400m/download_data.sh
```

At the same time, do our conversion and uploading which uses MDS (you will want to run them at the same time, or disk usage can get excessive):

```
./streaming/text/convert/laion/laion400m/convert_and_upload.sh
```

Monitor network i/o: `bwm-ng`

Monitor CPU usage: `htop`

Monitor disk i/o: `iotop`

Monitor disk usage: `df -h`
