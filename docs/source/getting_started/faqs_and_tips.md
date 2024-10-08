# 🤔 FAQs and Tips

## ❓ FAQs

### Can I write datasets in parallel? How does this work?
Yes, you can! Please see the [parallel dataset conversion](../preparing_datasets/parallel_dataset_conversion.ipynb) page for instructions. If you're using Spark, follow the [Spark dataframe to MDS](../preparing_datasets/spark_dataframe_to_mds.ipynb) example.

### Is StreamingDataset's `batch_size` the global or device batch size?
The `batch_size` argument to StreamingDataset is the *device* batch size. It should be set the same as the DataLoader `batch_size` argument. For optimal performance and deterministic resumption, you must pass `batch_size` to StreamingDataset.

### How can I calculate ingress and egress costs?
Ingress costs will depend on your GPU provider, but egress costs from cloud storage are equal to the egress costs for a single epoch of training. Streaming is smart about how samples are partitioned, and minimizes duplicate shard downloads between nodes. The egress cost is calculated as:

$$\text{Egress cost} = (\text{Egress cost per MB}) \times (\text{Average shard size in MB}) \times (\text{Total number of shards})$$

For multi-epoch training, if your nodes have persistent storage or if your training job does not experience hardware failures, the egress cost will be the same as a single epoch of training. Otherwise, with ephemeral storage and training failures, you will likely have to redownload shards.

### How can I mix and weight different data sources?
Mixing data sources is easy, flexible, and can even be controlled at the batch level. The [mixing data sources](../dataset_configuration/mixing_data_sources.md) page shows how you can do this.

### Can I use only a subset of a data source when training for multiple epochs?
Yes, you can! For example, if your dataset is 1000 samples, but you want to train only on 400 samples per epoch, simply set
`epoch` size to 400. For more control over how these 400 samples are chosen in each epoch, see the [inter-epoch sampling](../dataset_configuration/replication_and_sampling.md#inter-epoch-sampling) section.

### How can I apply a transformation to each sample?
StreamingDataset is a subclass of PyTorch's IterableDataset, so applying transforms works the exact same way. See [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) for an example on how to use transforms with PyTorch. Our [CIFAR-10 guide](../how_to_guides/cifar10.ipynb) also has an example of using transforms with StreamingDataset.

### If my dataset is larger than disk, how can I train?
You can set the per-node cache limit using StreamingDataset's `cache_limit` argument, detailed [here](../dataset_configuration/shard_retrieval.md#cache-limit). When shard usage hits the `cache_limit` Streaming will begin evicting shards.

### I'm seeing loss spikes and divergence on my training runs. How do I fix this?
Training loss may suffer from loss spikes or divergence for a variety of reasons. Higher quality shuffling and dataset mixing can help mitigate loss variance, divergence, and spikes. First, make sure that `shuffle` is set to `True` in your dataset. If you're already shuffling, you should make your shuffle strength higher. If using a shuffle-block-based shuffling algorithm like [`'py1e'`](../dataset_configuration/shuffling.md#py1e-default), [`'py1br'`](../dataset_configuration/shuffling.md#py1br), or [`'py1b'`](../dataset_configuration/shuffling.md#py1b), increase the `shuffle_block_size` parameter. If using an intra-shard shuffle such as [`'py1s'`](../dataset_configuration/shuffling.md#py1s) or [`'py2s'`](../dataset_configuration/shuffling.md#py2s), increase the `num_canonical_nodes` parameter. Read more about shuffling [here](../dataset_configuration/shuffling.md).

Changing how datasets are mixed can also help with training stability. Specifically, setting `batching_method` to `stratified` when mixing datasets provides consistent dataset mixing in every batch. Read more about dataset mixing [here](../dataset_configuration/mixing_data_sources.md).

### When training for multiple epochs, training takes a long time between epochs. How can I address this?
Training is likely taking longer between epochs due to DataLoader workers not persisting. Make sure to set `persistent_workers=True` in your DataLoader, which will keep `StreamingDataset` instances alive between epochs. More information can be found [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

If this still does not address the issue, refer to the [performance tuning page](../distributed_training/performance_tuning.md).

### I'm not seeing deterministic resumption on my training runs. How can I enable this?
To enable elastic determinism and resumption, you should be using the {class}`streaming.StreamingDataLoader` instead of the generic PyTorch DataLoader. You should also make sure you're passing in `batch_size` to StreamingDataset in addition to your DataLoader. Certain launchers, such as [Composer](https://github.com/mosaicml/composer), support deterministic resumption with StreamingDataset automatically. See the [resumption](../distributed_training/fast_resumption.md) page for more information.

### Is it possible for each global or device batch to consist only of samples from one Stream?
Yes. For global batches drawn from a single stream, use the `per_stream` batching method, and for device batches drawn from a single stream, use the `device_per_stream` batching method. More details are in the [batching methods](../dataset_configuration/mixing_data_sources.md#batching-methods) section.

### I'm seeing a shared memory error. How can I fix this?
Streaming uses shared memory to communicate between workers. These errors are indicative of stale shared memory, likely from a previous training run. To fix this, call `python` in your terminal and run the commands below:
<!--pytest.mark.skip-->
```
>>> import streaming.base.util as util
>>> util.clean_stale_shared_memory()
```

### What's the difference between StreamingDataset's `epoch_size`, `__len__()`, and `size()`?
The `epoch_size` attribute of StreamingDataset is the number of samples per epoch of training. The `__len__()` method returns the `epoch_size` divided by the number of devices -- it is the number of samples seen per device, per epoch. The `size()` method returns the number of unique samples in the underlying dataset. Due to upsampling/downsampling, `size()` may not be the same as `epoch_size`.

### What's the difference between `StreamingDataset` vs. datasets vs. streams?
`StreamingDataset` is the dataset class. It can take in multiple streams, which are just data sources. It combines these streams into a single dataset. `StreamingDataset` does not *stream* data, as continuous bytes; instead, it downloads shard files to enable a continuous flow of samples into the training job. `StreamingDataset` is an `IterableDataset` as opposed to a map-style dataset -- samples are retrieved as needed.


## 🤓 Helpful Tips

### Using locally available datasets
If your dataset is locally accessible from your GPUs, you only need to specify the `local` argument to StreamingDataset as the path to those shard files. You should leave the `remote` field as `None`.

### Access specific shards and samples
You can use the `get_item` method of StreamingDataset to access particular samples -- StreamingDataset supports NumPy-style indexing. To further access information at the shard and sample level, the StreamingDataset attributes below are useful:

- `dataset.stream_per_shard`: contains the stream index for each shard.
- `dataset.shards_per_stream`: contains the number of shards per stream
- `dataset.samples_per_shard`: contains the number of samples per shard
- `dataset.samples_per_stream`: contains the number of samples per stream
- `dataset.spanner`: maps global sample index to the corresponding shard index and relative sample index
- `dataset.shard_offset_per_stream`: contains the offset of the shard indices for a stream. Can be used to get the shard index in a certain stream from the global shard index.
- `dataset.prepare_shard(shard_id)`: downloads and extracts samples from shard with `shard_id`
- `dataset[sample_id]`: retrieves sample with `sample_id`, implicitly downloading the relevant shard.

You can use these in a variety of ways to inspect your dataset. For example, to retrieve the stream index, relative shard index in that stream, and sample index in that shard, for every sample in your dataset, you could do:
<!--pytest.mark.skip-->
```python
# Instantiate a StreamingDataset however you would like
dataset = StreamingDataset(
    ...
)
# Retrieves the number of unique samples -- no up or down sampling applied
num_dataset_samples = dataset.size
# Will contain tuples of (stream id, shard id, sample id)
stream_shard_sample_ids = []
for global_sample_idx in range(num_dataset_samples):
    # Go from global sample index -> global shard index and relative sample index (in the shard)
    global_shard_idx, relative_sample_idx = dataset.spanner[global_sample_idx]
    # Get the stream index of that shard
    stream_idx = dataset.stream_per_shard[global_shard_idx]
    # Get the relative shard index (in the stream) by subtracting the offset
    relative_shard_idx = global_shard_idx - dataset.shard_offset_per_stream[stream_idx]

    stream_shard_sample_ids.append((stream_idx, relative_shard_idx, relative_sample_idx))
```

### Don't make your shard file size too large or small
You can control the maximum file size of your shards with the `size_limit` argument to the `Writer` objects -- for example, in {class}`streaming.MDSWriter`. The default shard size is 67MB, and we see that 50-100MB shards work well across modalities and workloads. If shards are too small, then you will get too many download requests, and if shards are too large, then shard downloads become more expensive and harder to balance.
