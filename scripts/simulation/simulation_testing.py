import wandb
import matplotlib.pyplot as plt
import numpy as np
import yaml
from simulation_funcs import simulate


run_ids = ["py1e-testing/1pid105q"]

# common parameters
shards = 20850
samples_per_shard = 4093
avg_shard_size = 67092639
compressed_shard_size = 16000000
compression_ratio = compressed_shard_size/avg_shard_size
epochs = 1
avg_batch_time = 0.28
node_network_bandwidth = 1e8
throughput_window = 10


api = wandb.Api()

for run_id in run_ids:

    run = api.run("mosaic-ml/"+run_id)
    summary = run.summary
    config = run.config

    # get parameters from run config and summary
    batches_per_epoch = summary['_step']
    devices = int(config["num_gpus_per_node"])
    physical_nodes = int(config['n_gpus']/devices)
    # device_batch_size set for each run
    device_batch_size = int(config['global_train_batch_size']/(physical_nodes*devices))
    canonical_nodes = int(config['num_canonical_nodes'])
    workers = int(config["train_loader"]["num_workers"])
    predownload = int(config["train_loader"]["dataset"]["predownload"])
    cache_limit = None
    if "cache_limit" in config["train_loader"]["dataset"]:
        cache_limit = int(config["train_loader"]["dataset"]["cache_limit"])
    shuffle_algo = config["train_loader"]["dataset"]["shuffle_algo"]
    shuffle_block_size = int(config["train_loader"]["dataset"]["shuffle_block_size"])
    seed = config['seed']

    print(yaml.dump(config, default_flow_style=False))

    # get real throughput and network use from the run
    real_batch_throughput = run.history(samples=batches_per_epoch-throughput_window, keys=['throughput/batches_per_sec'], pandas=True)
    real_network_use = run.history(stream="system", pandas=True)['system.network.recv']

    # simulate throughput and network use given the inputs

    step_times, shard_downloads = simulate(shards, samples_per_shard, avg_shard_size,
                                       device_batch_size, avg_batch_time, batches_per_epoch,
                                       epochs, physical_nodes, devices, node_network_bandwidth,
                                       workers, canonical_nodes, predownload, cache_limit,
                                       shuffle_algo, shuffle_block_size, seed)
    
    immediate_batch_throughput = 1 / step_times

    shard_downloads_cumulative = np.cumsum(shard_downloads)

    step_times_rolling_avg = np.convolve(step_times, np.ones(throughput_window) / throughput_window, mode='valid')
    batch_throughput_rolling_avg = 1 / step_times_rolling_avg
    batch_throughput_rolling_avg = np.concatenate((np.array([0] * 9), batch_throughput_rolling_avg))

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title("throughput")
    #ax1.plot(real_batch_throughput["_step"], real_batch_throughput["throughput/batches_per_sec"], color="red", label="real")
    ax1.plot(np.arange(batch_throughput_rolling_avg.shape[0]), batch_throughput_rolling_avg, color="blue", label="sim")
    ax1.legend()

    ax2.set_title("network use")
    #ax2.plot(list(range(len(real_network_use))), real_network_use, color="red", label="real")
    ax2.plot(np.arange(shard_downloads_cumulative.shape[0]), shard_downloads_cumulative, color="blue", label="sim")
    ax2.legend()

    plt.show()