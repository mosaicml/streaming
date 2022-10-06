## English Wikipedia MDS dataset creation

### Training split:

- `python3 make_train_parallel.py`
  - calls `create_pretraining_data.py`
- `python3 merge_shard_groups.py`

### Evaluation split:

- `./make_eval.sh`
  - calls `create_pretraining_data.py`
  - calls `pick_eval_samples.py`
