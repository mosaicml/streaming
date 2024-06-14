python3 scripts/iteration/generate_parquet.py \
    --dataset data/iteration/parquet/

python3 scripts/iteration/parquet_to_lance.py \
    --parquet data/iteration/parquet/val/ \
    --lance data/iteration/lance_1024/val/ \
    --max_rows_per_group 1024

python3 scripts/iteration/parquet_to_streaming.py \
    --dataset data/iteration/parquet/val/

python3 scripts/iteration/bench.py \
    --streaming_dataset data/iteration/parquet/val/ \
    --lance_dataset data/iteration/lance_1024/val/ \
    --stats data/iteration/stats_1024_val.json

python3 scripts/iteration/plot.py \
    --stats data/iteration/stats_1024_val.json \
    --plot data/iteration/plot_1024_val.png
