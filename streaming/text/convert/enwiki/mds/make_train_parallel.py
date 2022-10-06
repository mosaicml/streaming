import os

num_cpus = os.cpu_count()
num_shards = 500
input_file_pattern = '/tmp/enwiki_preproc/results4/part-%05d-of-00500'
output_dir_pattern = '/tmp/mds-enwiki/train/group-%03d/'

for cpu in range(num_cpus):
    begin = cpu * num_shards // num_cpus
    end = (cpu + 1) * num_shards // num_cpus
    input_files = ','.join([input_file_pattern % i for i in range(begin, end)])
    output_dir = output_dir_pattern % cpu
    command = f'''
        python3 create_pretraining_data.py \
           --input_file {input_files} \
           --output_dir {output_dir} \
           --compression zstd:16 \
           --hashes sha1,xxh3_64 \
           --size_limit 134217728 \
           --vocab_file vocab.txt \
           --do_lower_case True \
           --max_seq_length 512 \
           --max_predictions_per_seq 76 \
           --masked_lm_prob 0.15 \
           --random_seed 12345 \
           --dupe_factor 10 &
    '''
    os.system(command)
