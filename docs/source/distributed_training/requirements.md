# Requirements for Distributed Training

Streaming is purpose built for fast, large-scale distributed training. It relies on the environment variables below, that must be set on each device/GPU to correctly assign data.

- **WORLD_SIZE**: Total number of processes to launch across all nodes.
- **LOCAL_WORLD_SIZE**: Total number of processes to launch for each node.
- **RANK**: Rank of the current process, which is the range between `0` to `WORLD_SIZE - 1`.
- **MASTER_ADDR**: The hostname for the rank-zero process.
- **MASTER_PORT**: The port for the rank-zero process.

Some launchers will automatically take care of setting environment variables. For example, using [Composer](https://docs.mosaicml.com/projects/composer/en/stable/) in conjunction with [MosaicML Platform](https://docs.mosaicml.com/projects/mcli/en/latest/) will automatically enable distributed training.

More info about using different distributed training launchers with Streaming can be found [here](using_launchers.md).

## Parallelism Strategies

Streaming supports a variety of distributed training parallelism strategies, including Distributed Data Parallelism ([DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)), Fully Sharded Data Parallelism ([FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), akin to [ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)), Hybrid Sharded Data Parallelism ([HSDP](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)), Tensor Parallelism ([TP](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-tensor-parallelism.html)), and Sequence Parallelism ([SP](https://arxiv.org/pdf/2105.13120.pdf)).

### Data Parallel strategies

Parallelism strategies like DDP, FSDP, and HSDP are all data-parallel strategies, where each device needs to see a unique part of the global training batch. StreamingDataset supports this out-of-the-box without any configuration changes.

### Data Replication strategies

Parallelism strategies like TP and SP require multiple devices to receive the same data samples, requiring replication. Simply set the `replication` argument to StreamingDataset to specify how many consecutive devices should receive the same data. An example can be found [here](../dataset_configuration/replication_and_sampling.md#replication).
