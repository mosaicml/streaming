# Streaming Simulator
A simulator for throughput, network use, and shuffle quality with MosaicML Streaming. The simulator allows you to:
- Plan runs and anticipate issues beforehand
- Find optimal run configurations
- Debug issues with underperforming runs
- Better understand the impact of different configurations

## Getting Started
Run the following to install simulator-specific dependencies, if they don't already exist:
```
pip install --upgrade "mosaicml-streaming[simulator]"
```
Then, simply run `simulator` in your command line to open the Web UI and get simulating!
## Key Features

### Throughput
Throughput is estimated for the duration of the run and is displayed as the simulation progresses. We estimate throughput by iterating over the samples of the dataset in order, and performing shard downloads based on an estimate of network bandwidth. The 10-step rolling average is displayed.

<img src="../_static/images/throughput.png" alt="Throughput Graph" width="500"/>

### Network Downloads
Cumulative network downloads are also estimated for the run and displayed. It is calculated in conjunction with throughput. If shards are compressed, we assume they are downloaded in compressed form and immediately uncompressed.

<img src="../_static/images/downloads.png" alt="Downloads Graph" width="500"/>

### Simulation Stats
We also provide various useful statistics from the simulation, such as:
- Minimum cache limit (i.e., maximum space used by live shards)
- Steps slowed down by shard downloads
- Estimated time to first batch
- Estimated warmup time (i.e., time until throughput maximized)

<img src="../_static/images/stats.png" alt="Simulation Stats" width="500"/>

### Shuffle Quality
You can choose to evaluate the quality of different shuffling algorithms for your run. We provide an estimate of shuffle quality based on the entropy calculated over the probability distribution of differences between neighboring sample indices and shard indices of the dataset. *These shuffle quality metrics are noisy and may not reflect the true strength of a shuffle.*

<img src="../_static/images/shuffle_quality_toggle.png" alt="Shuffle Quality Toggle" width="300"/>

<img src="../_static/images/shuffle_quality_graph.png" alt="Shuffle Quality Graph" width="500"/>

### Yaml Support
Yaml files that follow MosaicML conventions can be uploaded and simulated as well. Simply click the toggle, enter any needed additional information, and see your results. Parameters can also be modified to quickly test out configurations.

<img src="../_static/images/yaml_toggle.png" alt="Yaml Quality Toggle" width="300"/>
