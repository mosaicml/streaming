# Environments

StreamingDataset relies on certain environment variables that need to be set to work for the distributed workload. If the launcher that you are using does not set the below environment variables, you need to set it manually either in your script or export globally.

- **WORLD_SIZE**: Total number of processes to launch across all nodes.
- **LOCAL_WORLD_SIZE**: Total number of processes to launch for each node.
- **RANK**: Rank of the current process, which is the range between `0` to `WORLD_SIZE - 1`.
- **MASTER_ADDR**: The hostname for the rank-zero process.
- **MASTER_PORT**: The port for the rank-zero process.
