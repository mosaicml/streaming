#!/bin/sh

wandb login

img2dataset \
    --url_list laion400m-meta \
    --input_format parquet \
    --url_col URL \
    --caption_col TEXT \
    --output_format parquet \
    --output_folder laion400m-data \
    --processes_count 16 \
    --thread_count 128 \
    --image_size 256 \
    --save_additional_columns '["NSFW","similarity","LICENSE"]' \
    --enable_wandb True

touch laion400m-data/done
