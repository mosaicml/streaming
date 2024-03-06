# Shard Retrieval

Shards are downloaded on the fly during training and samples are retrieved from them. You can configure {class}`streaming.StreamingDataset`'s shard retrieval to meet your training job's needs. For more information about shard retrieval during distributed model training, refer to the [main concepts](../getting_started/main_concepts.md#Distributed-model-training) page.

## Loading datasets

### Pointing to your dataset

To train on a dataset that lives in a remote location, simply pass the path to StreamingDataset's `remote` argument. The dataset's `index.json` file should live at this directory. StreamingDataset works with all major cloud providers. The `local` argument should be used to specify where the downloaded shards will be stored, on local disk.

```python
dataset = StreamingDataset(
    remote = 's3://some-bucket/my-dataset',    # dataset lives at this remote path
    local = '/local/dataset',    # shards downloaded and stored locally at this path
)
```

If your dataset is already available on local disk for your GPUs to access, only specify the `local` argument.

```python
dataset = StreamingDataset(
    local = '/local/dataset',    # dataset shards are already locally available at this path
)
```

The `split` argument can be used to specify a particular subdirectory to use -- for example, a training dataset split.

```python
dataset = StreamingDataset(
    remote = 's3://some-bucket/my-dataset',
    local = '/local/dataset',
    split = 'train',    # dataset will be loaded from 's3://some-bucket/my-dataset/train'
)
```

### Multiple streams

If using multiple data sources, specify the `remote` and/or `local` paths for each one in a separate {class}`streaming.Stream` object, and pass those to StreamingDataset's `streams` argument. An example can be found [here](../getting_started/main_concepts.md#Remote-data-streams).

### Hash Validation

If you wrote out your dataset shards with specific hash functions (see [here](../preparing_datasets/basic_dataset_conversion.md#Configuring-dataset-writing)) and want to validate them at training time, set the `validate_hash` argument to StreamingDataset. Depending on the hash function, this may slow down data loading.

```python
dataset = StreamingDataset(
    ...
    validate_hash = 'sha1',    # validate shard using sha1 hash function
    ...
)
```

## Controlling shard downloads

### Downloading ahead

Setting the `predownload` argument ensures that StreamingDataset will download the shards needed for the upcoming `predownload` samples, per worker. For example, if `predownload` is set to 8, then each DataLoader worker will download the shards needed for up to 8 samples ahead of the current point in training. The default value of `predownload` in StreamingDataset performs well, so only set this argument if you want to prepare more samples ahead of the current training batch.

```python
dataset = StreamingDataset(
    ...
    predownload = 8,    # each worker will download shards for up to 8 samples ahead
    ...
)
```

### Retries and Timeout

Set the `download_retry` argument to the number of times a shard download should be retried. The `download_timeout` argument specifies, in seconds, how long to wait for a shard download before throwing an exception. For larger shards, a longer `download_timeout` can be necessary.

```python
dataset = StreamingDataset(
    ...
    download_retry = 3,    # retry shard downloads up to 3 times
    download_timeout = 120,    # wait 2 minutes for a shard to download
    ...
)
```

## Configure shard storage

### Cache limit

If you have limited local disk space, specify the `cache_limit` argument. Once locally stored shards reach the `cache_limit`, Streaming will begin evicting shards to stay under the limit. This is particularly useful for very large datasets or small disks. Setting `cache_limit` too low will hinder performance, since shards may be continually evicted and redownloaded. This can be specified as integer bytes or as a human-readable string.

```python
cache_limit = 10*1024**2    # cache limit of 10mb
cache_limit = '10mb'    # also a cache limit of 10mb
```

### Keeping compressed shards

If your dataset shards are compressed (see [here](../preparing_datasets/basic_dataset_conversion.md#Configuring-dataset-writing)), StreamingDataset will uncompress them upon download for use in training. To control whether the compressed versions of shards are kept locally, use the `keep_zip` flag. This defaults to `False`, meaning that StreamingDataset will default to deleting compressed shards and only keeping uncompressed shards.

```python
dataset = StreamingDataset(
    ...
    keep_zip = True,    # keep compressed versions of shards locally
    ...
)
```

