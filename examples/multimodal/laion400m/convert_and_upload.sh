#!/bin/sh

REMOTE=$1

python3 -m examples.multimodal.laion400m.convert_and_upload \
    --local laion400m-data \
    --remote $REMOTE \
    --keep_parquet 0 \
    --keep_mds 0
