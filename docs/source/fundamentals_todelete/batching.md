# Batching

You can choose how batches are constructed by specifying the `batching_method` argument when instantiating `StreamingDataset`. Currently, this can take on one of three values:

- `'random'`: (default) Samples for each batch are chosen at random from input streams. While stream proportions hold in aggregate over the course of training, this batching method does not guarantee that stream proportions hold for each batch.
- `'stratified'`: Every single batch is divided up between streams in the same proportions. Unlike in the default case, stream proportions hold for every batch, unlike in the default case, where they hold only in aggregate.
- `'per_stream'`: Each batch has samples from just one stream. In aggregate over all batches, stream proportions still hold.
