# just do one of 500 shards for demonstration purposes.
python3 create_pretraining_data.py \
   --input_file=/tmp/enwiki_preproc/results4/part-00000-of-00500 \
   --output_file=/tmp/enwiki_preproc/tfrecord/train-part-00000-of-00500 \
   --vocab_file=vocab.txt \
   --do_lower_case=True \
   --max_seq_length=512 \
   --max_predictions_per_seq=76 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=10
