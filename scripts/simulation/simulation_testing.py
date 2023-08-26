# testing script for simulator
# compares experimental data from wandb with simulation data.

import wandb
import matplotlib.pyplot as plt
import numpy as np
import yaml
from simulation_funcs import simulate
import pandas as pd

api = wandb.Api()

project_id = "mosaic-ml/streaming-shuffling-algo"
project_runs = api.runs(path=project_id, per_page=300)
project_runs_list = [run.id for run in project_runs]
skip = 0


# C4 neox compressed from OCI parameters
shards = 20850
samples_per_shard = 4093
avg_shard_size = 67092639
compressed_shard_size = 16000000
compression_ratio = compressed_shard_size / avg_shard_size
epochs = 1
time_per_sample = 0.0175
node_network_bandwidth = 2e9
throughput_window = 10

def get_similarity_percentage(real, sim):
    real_copy = real.reshape(1, -1)
    sim_copy = sim.reshape(1, -1)
    merged = np.concatenate((real_copy, sim_copy), axis=0)
    similarities = np.abs(real-sim)/np.max(merged, axis=0)
    nanmean = np.nanmean(similarities)
    return 1 - nanmean

for run_id in project_runs_list[skip:]:

    run = api.run(f"{project_id}/{run_id}")

    print(run.name)

    summary = run.summary
    config = run.config

    if '_step' not in summary:
        print("skipping unsuccessful run")
        continue

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
        cache_limit = config["train_loader"]["dataset"]["cache_limit"]
    shuffle_algo = None
    if "shuffle_algo" in config["train_loader"]["dataset"]:
        shuffle_algo = config["train_loader"]["dataset"]["shuffle_algo"]
    shuffle_block_size = config["train_loader"]["dataset"]["shuffle_block_size"]
    seed = config['seed']

    # get step timestamps, real throughput, and network use from the run
    step_timestamps = run.history(samples=batches_per_epoch, keys=["_timestamp"], pandas=True)
    real_batch_throughput = run.history(samples=batches_per_epoch-throughput_window, keys=["throughput/batches_per_sec"], pandas=True)

    real_network_use = run.history(stream="system", pandas=True)[["_timestamp", "system.network.recv"]]
    
    # merge real_network_use with step_timestamps
    merged_network_use = pd.merge_asof(real_network_use, step_timestamps, on="_timestamp", direction="nearest")

    # simulate throughput and network use given the inputs

    step_times, shard_downloads = simulate(shards, samples_per_shard, avg_shard_size,
                                       device_batch_size, time_per_sample, batches_per_epoch,
                                       epochs, physical_nodes, devices, node_network_bandwidth,
                                       workers, canonical_nodes, predownload, cache_limit,
                                       shuffle_algo, shuffle_block_size, seed)
    
    immediate_batch_throughput = 1 / step_times

    shard_downloads_cumulative = np.cumsum(shard_downloads)
    shard_downloads_steps = np.arange(shard_downloads.shape[0])
    sim_downloads = pd.DataFrame({"_step": shard_downloads_steps, "sim_downloads": shard_downloads_cumulative})
    # merge simulated downloads with real downloads dataframe
    merged_network_use = pd.merge_asof(merged_network_use, sim_downloads, on="_step", direction="nearest")

    step_times_rolling_avg = np.convolve(step_times, np.ones(throughput_window) / throughput_window, mode='valid')[:-1]
    batch_throughput_rolling_avg = 1 / step_times_rolling_avg
    sim_throughput = pd.DataFrame({"_step": throughput_window + np.arange(batch_throughput_rolling_avg.shape[0]), "sim_throughput": batch_throughput_rolling_avg})
    merged_throughput = pd.merge_asof(real_batch_throughput, sim_throughput, on="_step", direction="nearest")

    # get similarity scores
    throughput_similarity = get_similarity_percentage(merged_throughput["throughput/batches_per_sec"].to_numpy(), merged_throughput["sim_throughput"].to_numpy())
    network_similarity = get_similarity_percentage(physical_nodes*(merged_network_use["system.network.recv"].to_numpy()), compression_ratio*(merged_network_use["sim_downloads"].to_numpy()))

    # print params and results to easily paste to spreadsheet
    print(run.name, seed, canonical_nodes, physical_nodes, predownload, shuffle_algo, shuffle_block_size, cache_limit, batches_per_epoch, throughput_similarity, network_similarity)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title("throughput - score: " + str(throughput_similarity))
    ax1.plot(merged_throughput["_step"], merged_throughput["throughput/batches_per_sec"], color="red", label="real")
    ax1.plot(merged_throughput["_step"], merged_throughput["sim_throughput"], color="blue", label="sim")
    ax1.legend()

    ax2.set_title("network use - score: " + str(network_similarity))
    # wandb only logs network use for node 0. multiply by number of nodes to get total network use
    ax2.plot(merged_network_use["_timestamp"], physical_nodes*merged_network_use["system.network.recv"], color="red", label="real")
    # simulation assumes all shards are downloaded uncompressed (overestimates). multiply by compression ratio to get true network use
    ax2.plot(merged_network_use["_timestamp"], compression_ratio*merged_network_use["sim_downloads"], color="blue", label="sim")
    ax2.legend()

    fig.set_figheight(8)

    plt.show()