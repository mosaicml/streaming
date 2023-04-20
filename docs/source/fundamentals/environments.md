# Environments

StreamingDataset relies on certain environment variables that need to be set to work for the distributed workload.

- **WORLD_SIZE**: Total number of processes to launch across all nodes.
- **LOCAL_WORLD_SIZE**: Total number of processes to launch for each node.
- **RANK**: Rank of the current process, which is the range between `0` to `WORLD_SIZE - 1`.
