# Replication and Sampling

You can control how samples are replicated, chosen between epochs, and chosen from shards. These are useful for a variety of cases:
- **Replication**: Replicate training samples among subsets of devices. This is particularly useful for Tensor Parallelism (TP) or Sequence Parallelism (SP).
- **Inter-epoch Sampling**: Control if the samples seen across epochs should vary or not.
- **Sampling from shards**: Control how many samples to choose from each shard at a time.

Let's see when and how to use these features.

## Replication

Training with Tensor Parallelism (TP) or Sequence Parallelism (SP) requires multiple devices to see the same sample of data. The `replication` parameter of {class}`streaming.StreamingDataset`, controls how many consecutive devices will see the same samples in each batch. For example, if `replication` is set to 4 for a training job with 16 GPUs, devices 0 through 3 will see the same samples, devices 4 through 7 will see the same samples, and so on.

Be aware that samples are only replicated across consecutive GPUs, as denoted by their rank from [PyTorch's distributed module](https://pytorch.org/docs/stable/distributed.html).

## Inter-epoch sampling

You can choose how sampling from your dataset(s) occurs between epochs by specifying the `sampling_method` when instantiating `StreamingDataset`. This can be one of two values:

- `'balanced'`: (default) Samples are chosen at random from dataset(s) during each epoch.
- `'fixed'`: The same samples from the dataset(s) are chosen during every epoch.

For example, with `balanced` sampling, if the size of an epoch is 1000 samples, but my dataset contains 2000 samples, then each epoch will consist of 1000 samples taken at random from the underlying 2000. But with `fixed` sampling, the same 1000 samples that are seen in epoch 0 will be seen in all subsequent epochs as well.

## Sampling from shards

If all samples from a shard don't have to be used in training, the number of samples to choose from each shard is set by the `sampling_granularity` parameter to StreamingDataset. The `sampling_granularity` arg defaults to 1, meaning that one sample is chosen from each shard at a time.

This is particularly useful if just training on a small subset of your overall dataset. In this case, the way in which samples are chosen from shards becomes important, and directly impacts how many shards I have to download for the training job. For example, suppose the overall dataset has 10,000 samples, split up between 1000 shards of 100 samples each, but the epoch size is just 1000 samples. If `sampling_granularity` is set to 1, then the training dataset will consist of a single sample from each of the 1000 shards, meaning that all 1000 shards have to be downloaded over the course of the run. Instead, if `sampling_granularity` is set to 100, then the training dataset will consist of all 100 samples from just 10 shards, and only 10 shards will have to be downloaded for the run.

If the run's epoch size is large enough such that all shards have to be downloaded anyways, setting `sampling_granularity` will not change shard download demand.