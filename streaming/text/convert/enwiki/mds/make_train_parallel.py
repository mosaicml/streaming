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
    args.add_argument('--in_root',
                      type=str,
                      required=True,
                      help='Path to a valid results4/ directory containing the input data.')
    args.add_argument('--in_pattern',
                      type=str,
                      default='part-%05d-of-00500',
                      help='Input shard basename pattern.')
    args.add_argument('--in_num_shards',
                      type=int,
                      default=500,
                      help='Number of input shards.')
    args.add_argument('--out_root',
                      type=str,
                      required=True,
                      help='Output root directory containing shard dirs named like group-###.')
    args.add_argument('--out_pattern',
                      type=str,
                      default='group-%03d',
                      help='Output shard group dirname pattern.')
    args.add_argument('--compression',
                      type=str,
                      default='zstd:16',
                      help='Compression algorithm to use on output shards, if any. Default: ' +
                           'zstd:16.')
    args.add_argument('--hashes',
                      type=str,
                      default='sha1,xxh3_64',
                      help='Hash algorithms to use on output shards, if any. Default: ' +
                           'sha1,xxh3_64.')
    args.add_argument('--size_limit',
                      type=int,
                      default=1 << 27,
                      help='Shard size limit in bytes, after which point a new shard is ' +
                           'started. Default: 1 << 27.')
    args.add_argument('--vocab_file',
                      type=str,
                      default='vocab.txt',
                      help='The vocabulary file that the BERT model was trained on.')
    args.add_argument('--do_lower_case',
                      type=int,
                      default=1,
                      help='Whether to lower case the input text. Should be True for uncased ' +
                           'models and False for cased models.')
    args.add_argument('--max_seq_length',
                      type=int,
                      default=512,
                      help='Maximum sequence length.')
    args.add_argument('--max_predictions_per_seq',
                      type=int,
                      default=20,
                      help='Maximum number of masked LM predictions per sequence.')
    args.add_argument('--random_seed',
                      type=int,
                      default=12345,
                      help='Random seed for data generation.')
    args.add_argument('--dupe_factor',
                      type=int,
                      default=10,
                      help='Number of times to duplicate the input data (with different masks).')
    args.add_argument('--masked_lm_prob',
                      type=float,
                      default=0.15,
                      help='Masked LM probability.')
    args.add_argument('--short_seq_prob',
                      type=float,
                      default=0.1,
                      help='Probability of creating sequences which are shorter than the ' +
                           'maximum length.')
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
        input_files = ','.join([os.path.join(args.in_root, args.in_pattern % i)
                                for i in range(begin, end)])
        output_dir = os.path.join(args.out_root, args.out_pattern % cpu)
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
