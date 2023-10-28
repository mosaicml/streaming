## English Wikipedia MDS dataset creation

### Setup:

- Install tensorflow: `pip3 install tensorflow`
- Download `results_text.zip`: [Google Drive](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v)

### Training split creation:

- `python3 make_train_parallel.py`
  - calls `create_pretraining_data.py`
- `python3 merge_shard_groups.py`

### Evaluation split creation:

- `./make_eval.sh`
  - calls `create_pretraining_data.py`
  - calls `pick_eval_samples.py`
