# Shuffling

Shuffling is not simple because very large numbers of samples cannot be shuffled on the fly in their entirety with acceptable performance, and you would not want to if you could for distributed download reasons. Instead, we rely on a combination of factors to cleverly achieve the effect of a global shuffle while being shard-efficient.

## StreamingDataset arguments

StreamingDataset takes four arguments to directly control shuffling.

| Parameter | Type | Description |
| :-------- | :--- | :---------- |
| `shuffle` | `bool = False` | turn shuffling on or off |
| `shuffle_algo` | `str = 'py1b'` | which shuffling algorithm to use |
| `shuffle_seed` | `int = 9176` | all randomness in StreamingDataset is derived from this argument |
| `shuffle_block_size` | `int = 1 << 18` | shuffling unit used by py1b algorithm |

StreamingDataset also takes two other arguments that shuffling interacts with:

| Parameter | Type | Description |
| :-------- | :--- | :---------- |
| `predownload` | `Optional[int] = 100_000` | tune together with shuffle block size to keep workers from ever starving of shard pre-downloads while iterating (none means no limit) |
| `num_canonical_nodes` | `Optional[int] = None` | number of divisions of the sample space, which are iterated from beginning to end concurrently (defaults to number of initial physical nodes) |

## Algorithms

For `shuffle_algo`, you have four possible options, which have different tradeoffs. Once written, they cannot be changed, although it is easy to add new algorithms:

### naive

Globally shuffle the samples.

Useful for single-node training on small data, where you want the most random shuffle possible.

Statistically, this algorithm will result in all nodes downloading all shards, with those downloads all happening at the start of the epoch and needing to stay resident to make progress, bringing training to a crawl if the dataset is too large.

### py1b

Globally shuffle shards, divide that sample space over canonical nodes, then shuffle samples in fixed-size blocks (given by `shuffle_block_size`). So named because it shuffles samples in python, once, intra-block.

Shuffle block size should be set larger or much larger than a single shard. If so, this algorithm is useful for spacing out the contents of shards to mitigate a bad or non-existent pre-shuffle (i.e. if samples from the same shard are related in some way).

This algorithm requires more shards to be downloaded and stay resident to make progress than py1s or py2s, noticed as longer start/resume latency, as a multiple of shuffle block size divided by samples per shard. If you see step-like burstiness in throughput, your workers may not be downloading far enough ahead – try raising predownload (it should be scaled with block size). Step size scales with shuffle block size.

### py1s

Globally shuffle shards, divide that sample space over canonical nodes, then shuffle the samples within each shard or shard part. So named because it shuffles samples in python, once, intra-shard.

This algorithm only requires one shard to be resident at a time per canonical node, so is smoother and more responsive than py1b, however your pre-shuffle should be good. Also note that different modalities have vastly different numbers of samples per shard, with downstream effects on shuffle quality (rule of thumb: 500 samples/shard for vision, 50K samples/shard for text).

Shuffles twice as fast as py2s by being deterministic (“biased”) about assigning samples to canonical node divisions at boundary shards. In practice, we have not observed any negative downstream impacts from cutting a corner in this way. What effect does exist would be the strongest when your number of shards is very low relative to your number of canonical nodes.

### py2s

Globally shuffle shards, then shuffle the samples within each shard, then divide that sample space over canonical nodes, then shuffle the samples within each shard or shard part. So named because it shuffles samples in python, twice, intra-shard.

This algorithm only requires one shard to be resident at a time per canonical node, so is smoother and more responsive at downloading than py1b, however your pre-shuffle should be good. Also note that different modalities have vastly different numbers of samples per shard, with downstream effects on shuffle quality (rule of thumb: 500 samples/shard for vision, 50K samples/shard for text).

Shuffles roughly twice as slowly as py1s by being random (“correct”) about assigning samples to canonical node divisions at boundary shards. This becomes a pain point at around a billion samples.

## Factors to consider

Philosophers have long debated what it means to be a good shuffle. StreamingDataset relies on five approaches to shuffle quality which work synergistically:

### Pre-shuffle

The foundation of shuffle quality and therefore model learning is the preprocessing that was applied to the dataset, including deduping and pre-shuffling. Pre-shuffling refers to an offline preprocessing step that bakes in a global shuffle of the samples. You pre-shuffle once and benefit or suffer from the results forever.

For performance reasons, samples which are collocated in the same shard are much more likely to be seen in time proximity to one another. The choice of shard size also matters: generally, shards are shuffled globally but samples only intra-shard or intra-block. While there are mitigations below, it is important for balance that we get a good diversity of samples on each shard and minimize repetition.

### Shuffle algorithm

How the shuffle works intimately impacts the distribution of samples and weaknesses thereof.  See the preceding section for shuffling algorithms.

### Shuffle block size

You can strengthen the shuffle by increasing the size of the shuffling unit, within reason. For py1s or py2s this is the shard, but py1b provides a sliding scale via `shuffle_block_size` all the way from one sample at a time to all the samples at once (which would be like naive but with canonical node divisions).

Large shuffle block sizes can save you from a bad or missing pre-shuffle. They are also a proportionally cheap and essential measure to take when training for many epochs on small datasets with very few devices. Conversely, large shuffle block sizes are a superfluous waste of time if training with many canonical nodes and many devices on many shards. There is a balance.

### Number of canonical nodes

When iterating, the sample space is divided evenly according to the number of canonical nodes. These divisions are read concurrently from beginning to end striping over dataloader workers in a precise pattern that preserves elastic determinism.

The higher that `num_canonical_nodes` is set, the more independent non-overlapping paths the StreamingDataset replicas take through the shards per model replica (increasing data source diversity), and the more shards need to be downloaded concurrently. Data source diversity becomes increasingly important as you raise the number of different streams comprising the dataset. `num_canonical_nodes` can be raised arbitrarily high so long as the number of physical nodes evenly divides into it, which is ultimately limited by download throughput.

### Splitting shard repeats

The final shuffle quality technique is so important, it is always turned on. When upsampling, each repeat of a shard, including the last fractional repeat if it exists, is treated as a different shard for the purposes of shuffling. This results in the copies getting scattered across the epoch’s sample space, at the cost of more downloads.

Without this, StreamingDataset would have to up/down-sample by stretching shards larger or smaller. Heavily upsampling shards would cause the model to see the same samples many times in rapid succession (at scale), which we have found interacts disastrously with small shuffle units, modulo data augmentation. A potential landmine during training.

Our general advice on shuffling is that there are different tradeoffs at play, and the best answer often depends. We endeavor to provide reasonable defaults. Shuffling choices 2-4 can and should be tested empirically on your own models and your own data.
