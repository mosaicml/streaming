# 🤔 FAQs and Tips

## ❓ FAQs

### Can I write datasets in parallel? How does this work?
Yes, you can! Please see the [parallel dataset conversion](../preparing_datasets/parallel_dataset_conversion.md) page for instructions. If you're using Spark, follow the [Spark dataframe to MDS](../preparing_datasets/spark_dataframe_to_mds.md) example.

### How can I calculate ingress and egress costs?
Ingress costs will depend on your GPU provider, but egress costs from cloud storage are equal to the egress costs for a single epoch of training. Streaming is smart about how samples are partitioned, and minimizes duplicate shard downloads between nodes. The egress cost is calculated as:

$$C = R\cdot S \cdot N$$

Where $C$ is the egress cost, $R$ is the egress cost per MB, $S$ is the average shard size, in MB, and $N$ is the total number of shard files.

For multi-epoch training, if your nodes have persistent storage or if your training job does not experience hardware failures, the egress cost will be the same as a single epoch of training. Otherwise, with ephemeral storage and training failures, you will likely have to redownload shards.

### How can I mix different data sources?
Mixing data sources is easy, flexible, and can even be controlled at the batch level as well. The [mixing data sources](../dataset_configuration/mixing_data_sources.md) page shows how you can do this.

### I'm seeing loss spikes and divergence on my training runs. How do I fix this?
Loss spikes and divergence issues can, at times, be related to your samples. First, make sure that `shuffle` is set to `True` in your dataset. If you're already shuffling, make your shuffle stronger by increasing the dataset's `shuffle_block_size` parameter, or change the `batching_method` parameter to `stratified` to ensure that your model always sees the same proportion of samples from different data sources. More information on shuffling can be found [here](../dataset_configuration/sampling_and_shuffling.md#Shuffling), and on batching methods [here](../dataset_configuration/mixing_data_sources.md#Batching-methods).

### I'm not seeing deterministic resumption on my training runs. How can I enable this?
To enable elastic determinism and resumption, you should be using the {class}`streaming.StreamingDataLoader` instead of the generic PyTorch DataLoader. You should also make sure you're passing in `batch_size` to StreamingDataset in addition to your DataLoader. Certain launchers, such as [Composer](https://github.com/mosaicml/composer), support deterministic resumption with StreamingDataset automatically. See the [resumption](../distributed_training/fast_resumption.md) page for more information.

### I'm seeing a shared memory error. How can I fix this?
Streaming uses shared memory to communicate between workers. These errors are indicative of stale shared memory, likely from a previous training run. To fix this, call `python` in your terminal and run the commands below:
```
>>> import streaming.base.util as util
>>> util.clean_stale_shared_memory()
```

### What's the difference between `StreamingDataset` vs. datasets vs. streams?
`StreamingDataset` is the dataset class. It can take in multiple streams, which are just data sources. It combines these streams into a single dataset. `StreamingDataset` does not *stream* data, as continuous bytes; instead, it downloads shard files to enable a continuous flow of samples into the training job. `StreamingDataset` is an `IterableDataset` as opposed to a map-style dataset -- samples are retrieved as needed.


## Helpful Tips

### Using locally available datasets
If your dataset is locally accessible from your GPUs, you only need to specify the `local` argument to StreamingDataset as the path to those shard files. You should leave the `remote` field as `None`.

### Pass `batch_size` to StreamingDataset for deterministic training
To have deterministic training, you should pass the `batch_size` argument to StreamingDataset. This is the same batch size as your DataLoader uses. StreamingDataset needs this information to create the sample partition correctly -- see how the sample partition is created [here](main_concepts.md#Distributed-model-training). Read more about fast resumption [here](../distributed_training/fast_resumption.md).

### Don't make your shard file size too large or small
You can control the maximum file size of your shards with the `size_limit` argument to the `Writer` objects -- for example, in {class}`streaming.MDSWriter`. The default shard size is 67MB, and we see that 50-100MB shards work well across modalities and workloads. If shards are too small, then you will get too many download requests, and if shards are too large, then shard downloads become more expensive and harder to balance.