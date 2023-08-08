# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from streaming.base.converters.delta_to_mds import DeltaMdsConverter, default_mds_kwargs


class TestDeltaMdsConverter:

    def test_delta_mds_converter(self):
        """Test from databricks."""
        dmc = DeltaMdsConverter()

        remote = ''
        input_path = '/refinedweb/raw'
        mds_path = '/Volumes/datasets/default/mosaic_hackathon/mds/ml/refinedweb'

        dmc.execute(delta_parquet_path=input_path,
                    mds_path=mds_path,
                    partition_size=2048,
                    merge_index=True,
                    sample_ratio=-1,
                    remote=remote,
                    mds_kwargs=default_mds_kwargs)
