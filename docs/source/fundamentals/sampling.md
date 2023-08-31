# Sampling

You can choose how sampling from your dataset(s) occurs between epochs by specifying the `sampling_method` when instantiating `StreamingDataset`. Currently, this can take on one of two values:

- `'balanced'`: (default) Samples are chosen at random from dataset(s) during each epoch according to the proportions specified.
- `'fixed`: The same samples from the dataset(s) are chosen during every epoch, still according to the proportions specified.
- `'consistent_batch_composition'`: Every single global batch is consistently composed of samples from streams in the same proportions. Unlike in the default case, stream proportions hold even for each global batch, unlike in the default case, where they hold only in aggregate.
