python3 create_pretraining_data.py \
  --input_file=/tmp/enwiki_preproc/results4/eval.txt \
  --output_dir=/tmp/enwiki_preproc/mds/eval_intermediate/ \
  --compression=zstd:16 \
  --hashes=sha1,xxh3_64 \
  --size_limit=67108864 \
  --vocab_file=vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python3 pick_eval_samples.py \
  --in_root=/tmp/enwiki_preproc/mds/eval_intermediate/ \
  --out_root=/dataset/mds-enwiki/val/ \
  --num_examples_to_pick=10000
