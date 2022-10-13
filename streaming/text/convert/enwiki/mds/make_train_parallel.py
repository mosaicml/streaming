# Copyright 2022 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Invoke create_pretraining_data.py on shard ranges on all the cores."""

import os
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--in_pattern',
                      type=str,
                      default='/tmp/enwiki_preproc/results4/part-%05d-of-00500')
    args.add_argument('--in_num_shards', type=int, default=500)
    args.add_argument('--out_pattern', type=str, default='/tmp/mds-enwiki/train/group-%03d/')
    args.add_argument('--compression', type=str, default='zstd:16')
    args.add_argument('--hashes', type=str, default='sha1,xxh3_64')
    args.add_argument('--size_limit', type=int, default=1 << 27)
    args.add_argument('--vocab_file', type=str, default='vocab.txt')
    args.add_argument('--do_lower_case', type=int, default=1)
    args.add_argument('--max_seq_length', type=int, default=512)
    args.add_argument('--max_predictions_per_seq', type=int, default=76)
    args.add_argument('--random_seed', type=int, default=12345)
    args.add_argument('--dupe_factor', type=int, default=10)
    args.add_argument('--masked_lm_prob', type=float, default=0.15)
    args.add_argument('--short_seq_prob', type=float, default=0.1)
    return args.parse_args()


def main(args: Namespace) -> None:
    """Invoke create_pretraining_data.py on shard ranges in parallel.

    Args:
        args (Namespace): Command-line arguments.
    """
    num_cpus = os.cpu_count() or 1
    for cpu in range(num_cpus):
        begin = cpu * args.in_num_shards // num_cpus
        end = (cpu + 1) * args.in_num_shards // num_cpus
        input_files = ','.join([args.in_pattern % i for i in range(begin, end)])
        output_dir = args.out_pattern % cpu
        command = f'''
            python3 create_pretraining_data.py \
                --input_file {input_files} \
                --output_dir {output_dir} \
                --compression {args.compression} \
                --hashes {args.hashes} \
                --size_limit {args.size_limit} \
                --vocab_file {args.vocab_file} \
                --do_lower_case {'True' if args.do_lower_case else 'False'} \
                --max_seq_length {args.max_seq_length} \
                --max_predictions_per_seq {args.max_predictions_per_seq} \
                --random_seed {args.random_seed} \
                --dupe_factor {args.dupe_factor} \
                --masked_lm_prob {args.masked_lm_prob} \
                --short_seq_prob {args.short_seq_prob} &
        '''
        assert not os.system(command)


if __name__ == '__main__':
    main(parse_args())
