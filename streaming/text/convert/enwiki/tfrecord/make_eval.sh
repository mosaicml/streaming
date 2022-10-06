python3 create_pretraining_data.py \
  --input_file=/tmp/enwiki_preproc/results4/eval.txt \
  --output_file=/tmp/enwiki_preproc/tfrecord/eval_intermediate \
  --vocab_file=vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python3 pick_eval_samples.py \
  --input_tfrecord=/tmp/enwiki_preproc/tfrecord/eval_intermediate \
  --output_tfrecord=/tmp/enwiki_preproc/tfrecord/eval_10k \
  --num_examples_to_pick=10000
