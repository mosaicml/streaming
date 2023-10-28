from argparse import ArgumentParser
import os


def parse_args():
    args = ArgumentParser()
    args.add_argument('--input_pattern', type=str, required=True)
    args.add_argument('--output_pattern', type=str, required=True)
    args.add_argument('--num_shards', type=int, default=500)
    args.add_argument('--num_cpus', type=int, default=0)
    return args.parse_args()


def main(arg):
    num_cpus = args.num_cpus or os.cpu_count()
    for cpu in range(num_cpus):
        begin = cpu * args.num_shards // num_cpus
        end = (cpu + 1) * args.num_shards // num_cpus
        input_files = ','.join([args.input_pattern % i for i in range(begin, end)])
        output_file = args.output_pattern % cpu
        command = f'''
            python3 create_pretraining_data.py \
                --input_file {input_files} \
                --output_file {output_file} \
                --vocab_file vocab.txt \
                --do_lower_case True \
                --max_seq_length 512 \
                --max_predictions_per_seq 76 \
                --masked_lm_prob 0.15 \
                --random_seed 12345 \
                --dupe_factor 10 &
        '''
        os.system(command)


if __name__ == '__main__':
    main(parse_args())
