#!/bin/sh

if hash wandb 2> /dev/null; then
    wandb login
    ENABLE_WANDB=True
else
    ENABLE_WANDB=False
fi

img2dataset \
    --url_list laion400m-meta \
    --input_format parquet \
    --url_col URL \
    --caption_col TEXT \
    --output_format parquet \
    --output_folder laion400m-data \
    --processes_count 32 \
    --thread_count 128 \
    --image_size 256 \
    --save_additional_columns '["NSFW","similarity","LICENSE"]' \
    --enable_wandb $ENABLE_WANDB

touch laion400m-data/done
