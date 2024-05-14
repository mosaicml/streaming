# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
from typing import Any, Tuple

import numpy as np
import pytest

from streaming import CSVWriter, JSONWriter, MDSWriter, StreamingDataset, TSVWriter, XSVWriter
from tests.common.datasets import NumberAndSayDataset, SequenceDataset
from tests.common.utils import get_config_in_bytes

logger = logging.getLogger(__name__)


class TestMDSWriter:

    def test_invalid_args(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        with pytest.raises(ValueError, match=f'.*Invalid Writer argument.*'):
            _ = MDSWriter(out=local, columns=columns, min_workers=1)

    def test_max_size_limit(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        with pytest.raises(ValueError, match=f'`size_limit` must be less than*'):
            _ = MDSWriter(out=local, columns=columns, size_limit=2**32)

    @pytest.mark.parametrize('num_samples', [100])
    @pytest.mark.parametrize('size_limit', [32])
    def test_config(self, local_remote_dir: Tuple[str, str], num_samples: int,
                    size_limit: int) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(num_samples)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        expected_config = {
            'version': 2,
            'format': 'mds',
            'compression': None,
            'hashes': [],
            'size_limit': size_limit,
            'column_names': dataset.column_names,
            'column_encodings': dataset.column_encodings,
            'column_sizes': dataset.column_sizes
        }
        writer = MDSWriter(out=local,
                           columns=columns,
                           compression=None,
                           hashes=None,
                           size_limit=size_limit)
        assert writer.get_config() == expected_config

    @pytest.mark.parametrize('num_samples', [1000, 10000])
    @pytest.mark.parametrize('size_limit', [4096, 16_777_216])
    def test_number_of_files(self, local_remote_dir: Tuple[str, str], num_samples: int,
                             size_limit: int) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(num_samples)

        columns = dict(zip(dataset.column_names, dataset.column_encodings))

        config_data_bytes = get_config_in_bytes('mds', size_limit, dataset.column_names,
                                                dataset.column_encodings, dataset.column_sizes)
        extra_bytes_per_shard = 4 + 4 + len(config_data_bytes)
        extra_bytes_per_sample = 4

        first_sample_body = dataset.get_sample_in_bytes(0)
        first_sample_body = list(first_sample_body.values())
        first_sample_head = np.array([
            len(data)
            for data, size in zip(first_sample_body, dataset.column_sizes)
            if size is None
        ],
                                     dtype=np.uint32)
        first_sample_bytes = len(first_sample_head.tobytes() +
                                 b''.join(first_sample_body)) + extra_bytes_per_sample

        expected_samples_per_shard = (size_limit - extra_bytes_per_shard) // first_sample_bytes
        expected_num_shards = math.ceil(num_samples / expected_samples_per_shard)
        expected_num_files = expected_num_shards + 1  # index file and compression metadata file

        with MDSWriter(out=local,
                       columns=columns,
                       compression=None,
                       hashes=None,
                       size_limit=size_limit) as out:
            for sample in dataset:
                out.write(sample)
        files = os.listdir(local)
        logger.info(f'Number of files: {len(files)}')

        assert len(files) == expected_num_files, \
            f'Files written ({len(files)}) != expected ({expected_num_files}).'

    @pytest.mark.parametrize('num_samples', [50000])
    @pytest.mark.parametrize('size_limit', [65_536])
    @pytest.mark.parametrize('seed', [1234])
    def test_dataset_iter_determinism(self, local_remote_dir: Tuple[str, str], num_samples: int,
                                      size_limit: int, seed: int) -> None:
        compression = 'zstd:7'
        hashes = ['sha1', 'xxh3_64']

        local, _ = local_remote_dir
        dataset = NumberAndSayDataset(num_samples, seed=seed)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        with MDSWriter(out=local,
                       columns=columns,
                       compression=compression,
                       hashes=hashes,
                       size_limit=size_limit) as out:
            for sample in dataset:
                out.write(sample)

        # Apply the seed again for numpy determinism
        dataset.seed = seed

        mds_dataset = StreamingDataset(local=local, shuffle=False, batch_size=1)
        # Ensure length of dataset is equal
        assert len(dataset) == len(mds_dataset) == num_samples

        # Ensure sample iterator is deterministic
        for before, after in zip(dataset, mds_dataset):
            assert before == after

    def test_exist_ok(self, local_remote_dir: Tuple[str, str]) -> None:
        num_samples = 1000
        size_limit = 4096
        local, _ = local_remote_dir
        dataset = SequenceDataset(num_samples)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))

        # Write entire dataset initially
        with MDSWriter(out=local, columns=columns, size_limit=size_limit) as out:
            for sample in dataset:
                out.write(sample)
        num_orig_files = len(os.listdir(local))

        # Write single sample with exist_ok set to True
        with MDSWriter(out=local, columns=columns, size_limit=size_limit, exist_ok=True) as out:
            out.write(dataset[0])
        num_files = len(os.listdir(local))

        # Two files for single sample (index.json and one shard)
        assert num_files == 2
        # Should be more files generated for the entire dataset, which are then deleted as exist_ok is True
        assert num_orig_files > num_files

        # Check exception is raised when exist_ok is False and local already exists
        with pytest.raises(FileExistsError, match='Directory is not empty'):
            with MDSWriter(out=local, columns=columns, size_limit=size_limit) as out:
                out.write(dataset[0])


class TestJSONWriter:

    def test_max_size_limit(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        with pytest.raises(ValueError, match=f'`size_limit` must be less than*'):
            _ = JSONWriter(out=local, columns=columns, size_limit=2**32)

    @pytest.mark.parametrize('num_samples', [100])
    @pytest.mark.parametrize('size_limit', [32])
    def test_config(self, local_remote_dir: Tuple[str, str], num_samples: int,
                    size_limit: int) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(num_samples)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        expected_config = {
            'version': 2,
            'format': 'json',
            'compression': None,
            'hashes': [],
            'size_limit': size_limit,
            'columns': columns,
            'newline': '\n'
        }
        writer = JSONWriter(out=local,
                            columns=columns,
                            compression=None,
                            hashes=None,
                            size_limit=size_limit)
        assert writer.get_config() == expected_config

    @pytest.mark.parametrize('num_samples', [50000])
    @pytest.mark.parametrize('size_limit', [65_536])
    @pytest.mark.parametrize('seed', [1234])
    def test_dataset_iter_determinism(self, local_remote_dir: Tuple[str, str], num_samples: int,
                                      size_limit: int, seed: int) -> None:
        compression = 'zstd:7'
        hashes = ['sha1', 'xxh3_64']

        local, _ = local_remote_dir
        dataset = NumberAndSayDataset(num_samples, seed=seed)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        with JSONWriter(out=local,
                        columns=columns,
                        compression=compression,
                        hashes=hashes,
                        size_limit=size_limit) as out:
            for sample in dataset:
                out.write(sample)

        # Apply the seed again for numpy determinism
        dataset.seed = seed

        mds_dataset = StreamingDataset(local=local, shuffle=False, batch_size=1)
        # Ensure length of dataset is equal
        assert len(dataset) == len(mds_dataset) == num_samples

        # Ensure sample iterator is deterministic
        for before, after in zip(dataset, mds_dataset):
            assert before == after

    def test_exist_ok(self, local_remote_dir: Tuple[str, str]) -> None:
        num_samples = 1000
        size_limit = 4096
        local, _ = local_remote_dir
        dataset = SequenceDataset(num_samples)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))

        # Write entire dataset initially
        with JSONWriter(out=local, columns=columns, size_limit=size_limit) as out:
            for sample in dataset:
                out.write(sample)
        num_orig_files = len(os.listdir(local))

        # Write single sample with exist_ok set to True
        with JSONWriter(out=local, columns=columns, size_limit=size_limit, exist_ok=True) as out:
            out.write(dataset[0])
        num_files = len(os.listdir(local))

        # Three files for single sample (index.json, one shard, and one shard metadata)
        assert num_files == 3
        # Should be more files generated for the entire dataset, which are then deleted as exist_ok is True
        assert num_orig_files > num_files

        # Check exception is raised when exist_ok is False and local already exists
        with pytest.raises(FileExistsError, match='Directory is not empty'):
            with JSONWriter(out=local, columns=columns, size_limit=size_limit) as out:
                out.write(dataset[0])


class TestXSVWriter:

    def test_max_size_limit(self, local_remote_dir: Tuple[str, str]):
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        separator = ','
        with pytest.raises(ValueError, match=f'`size_limit` must be less than*'):
            _ = XSVWriter(out=local, columns=columns, separator=separator, size_limit=2**32)

    @pytest.mark.parametrize('num_samples', [100])
    @pytest.mark.parametrize('size_limit', [32])
    @pytest.mark.parametrize(('writer', 'name'), [(XSVWriter, 'xsv'), (TSVWriter, 'tsv'),
                                                  (CSVWriter, 'csv')])
    def test_config(self, local_remote_dir: Tuple[str, str], num_samples: int, size_limit: int,
                    writer: Any, name: str) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(num_samples)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        expected_config = {
            'version': 2,
            'format': name,
            'compression': None,
            'hashes': [],
            'size_limit': size_limit,
            'column_names': dataset.column_names,
            'column_encodings': dataset.column_encodings,
            'newline': '\n'
        }
        if writer.__name__ == XSVWriter.__name__:
            separator = ','
            expected_config['separator'] = separator
            writer = writer(out=local,
                            columns=columns,
                            separator=separator,
                            compression=None,
                            hashes=None,
                            size_limit=size_limit)
        else:
            writer = writer(out=local,
                            columns=columns,
                            compression=None,
                            hashes=None,
                            size_limit=size_limit)
        assert writer.get_config() == expected_config

    @pytest.mark.parametrize('num_samples', [50000])
    @pytest.mark.parametrize('size_limit', [65_536])
    @pytest.mark.parametrize('seed', [1234])
    @pytest.mark.parametrize('writer', [XSVWriter, TSVWriter, CSVWriter])
    def test_dataset_iter_determinism(self, local_remote_dir: Tuple[str, str], num_samples: int,
                                      size_limit: int, seed: int, writer: Any) -> None:
        compression = 'zstd:7'
        hashes = ['sha1', 'xxh3_64']

        local, _ = local_remote_dir
        dataset = NumberAndSayDataset(num_samples, seed=seed)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        if writer.__name__ == XSVWriter.__name__:
            with writer(out=local,
                        columns=columns,
                        separator=',',
                        compression=compression,
                        hashes=hashes,
                        size_limit=size_limit) as out:
                for sample in dataset:
                    out.write(sample)
        else:
            with writer(out=local,
                        columns=columns,
                        compression=compression,
                        hashes=hashes,
                        size_limit=size_limit) as out:
                for sample in dataset:
                    out.write(sample)

        # Apply the seed again for numpy determinism
        dataset.seed = seed

        mds_dataset = StreamingDataset(local=local, shuffle=False, batch_size=1)
        # Ensure length of dataset is equal
        assert len(dataset) == len(mds_dataset) == num_samples

        # Ensure sample iterator is deterministic
        for before, after in zip(dataset, mds_dataset):
            assert before == after

    @pytest.mark.parametrize('writer', [XSVWriter, TSVWriter, CSVWriter])
    def test_exist_ok(self, local_remote_dir: Tuple[str, str], writer: Any) -> None:
        num_samples = 1000
        size_limit = 4096
        local, _ = local_remote_dir
        dataset = SequenceDataset(num_samples)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))

        # Write entire dataset initially
        if writer.__name__ == XSVWriter.__name__:
            with writer(out=local, columns=columns, size_limit=size_limit, separator=',') as out:
                for sample in dataset:
                    out.write(sample)
        else:
            with writer(out=local, columns=columns, size_limit=size_limit) as out:
                for sample in dataset:
                    out.write(sample)
        num_orig_files = len(os.listdir(local))

        # Write single sample with exist_ok set to True
        if writer.__name__ == XSVWriter.__name__:
            with writer(out=local,
                        columns=columns,
                        size_limit=size_limit,
                        separator=',',
                        exist_ok=True) as out:
                out.write(dataset[0])
        else:
            with writer(out=local, columns=columns, size_limit=size_limit, exist_ok=True) as out:
                out.write(dataset[0])
        num_files = len(os.listdir(local))

        # Three files for single sample (index.json, one shard, and one shard metadata)
        assert num_files == 3
        # Should be more files generated for the entire dataset, which are then deleted as exist_ok is True
        assert num_orig_files > num_files

        # Check exception is raised when exist_ok is False and local already exists
        with pytest.raises(FileExistsError, match='Directory is not empty'):
            if writer.__name__ == XSVWriter.__name__:
                with writer(out=local, columns=columns, size_limit=size_limit,
                            separator=',') as out:
                    out.write(dataset[0])
            else:
                with writer(out=local, columns=columns, size_limit=size_limit) as out:
                    out.write(dataset[0])
