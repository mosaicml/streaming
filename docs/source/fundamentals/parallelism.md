# Parallelism

Streaming supports data parallelism as well as sequence/tensor parallelism.

- **Data Parallelism**: Streaming supports this by default. Each device will get a unique portion of
each global batch. Samples are not replicated across devices. FSDP and HSDP both fall under this
category.
- **Sequence/Tensor Parallelism**: These parallelism strategies require groups of devices to share a
portion of each global batch. Specifying the `replication` argument of `StreamingDataset` to `x`
ensures that `x` consecutive devices will receive the same data. For example, `replication=4`
sends one set of samples to devices 0 through 3, another to devices 4 through 7, and so on.
