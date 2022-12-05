#!/bin/sh

python3 -m streaming.text.convert.laion.laion400m.convert_and_upload \
    --local laion400m-data \
    --remote s3://mosaicml-internal-dataset-laion/laion400m/mds/2/ \
    --keep_parquet 0 \
    --keep_mds 0
