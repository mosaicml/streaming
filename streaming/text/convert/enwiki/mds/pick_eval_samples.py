"""Script for picking certain number of samples."""

from argparse import ArgumentParser

from streaming import StreamingDataset, MDSWriter


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        '--in_root',
        type=str,
        required=True,
        help='Local directory path of the input raw dataset',
    )
    args.add_argument(
        '--out_root',
        type=str,
        required=True,
        help='Directory path to store the output dataset',
    )
    args.add_argument(
        '--compression',
        type=str,
        default='zstd:16',
        help='Compression algorithm to use. Default: zstd:16',
    )
    args.add_argument(
        '--hashes',
        type=str,
        default='sha1,xxh3_64',
        help='Hashing algorithms to apply to shard files. Default: sha1,xxh3_64',
    )
    args.add_argument(
        '--size_limit',
        type=int,
        default=1 << 26,
        help='Shard size limit, after which point to start a new shard. Default: 1 << 26',
    )
    args.add_argument(
        '--num_examples_to_pick',
        type=int,
        default=10000,
        help='Number of examples to select. Default: 10_000',)
    return args.parse_args()


def main(args):
    dataset = StreamingDataset(local=args.in_root, shuffle=False)
    columns = {
        'input_ids': 'bytes',
        'input_mask': 'bytes',
        'segment_ids': 'bytes',
        'masked_lm_positions': 'bytes',
        'masked_lm_ids': 'bytes',
        'masked_lm_weights': 'bytes',
        'next_sentence_labels': 'bytes',
    }
    hashes = args.hashes.split(',') if args.hashes else []
    with MDSWriter(out=args.out_root, 
                   columns=columns, 
                   compression=args.compression, 
                   hashes=hashes, 
                   size_limit=args.size_limit) as writer:
        pick_ratio = dataset.index.total_samples / args.num_examples_to_pick
        for i in range(args.num_examples_to_pick):
            sample = dataset[int(i * pick_ratio)]
            writer.write(sample)


if __name__ ==  '__main__':
    main(parse_args())
