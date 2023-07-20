# Copyright 2023 MosaicML Streaming authors
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

        mds_dataset = StreamingDataset(local=local, shuffle=False)
        # Ensure length of dataset is equal
        assert len(dataset) == len(mds_dataset) == num_samples

        # Ensure sample iterator is deterministic
        for before, after in zip(dataset, mds_dataset):
            assert before == after


class TestJSONWriter:

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

        mds_dataset = StreamingDataset(local=local, shuffle=False)
        # Ensure length of dataset is equal
        assert len(dataset) == len(mds_dataset) == num_samples

        # Ensure sample iterator is deterministic
        for before, after in zip(dataset, mds_dataset):
            assert before == after


class TestXSVWriter:

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

        mds_dataset = StreamingDataset(local=local, shuffle=False)
        # Ensure length of dataset is equal
        assert len(dataset) == len(mds_dataset) == num_samples

        # Ensure sample iterator is deterministic
        for before, after in zip(dataset, mds_dataset):
            assert before == after


class TestAllWriterSizeLimits:

    @pytest.mark.parametrize('input,expected', [('3Kb', 3 * 1024), ('4mb', 4 * 1024**2),
                                                ('2Gb', 2 * 1024**3), ('5tB', 5 * 1024**4),
                                                ('0.7pb', int(0.7 * 1024**5)),
                                                ('0.03EB', int(0.03 * 1024**6)),
                                                ('537Zb', 537 * 1024**7), ('91yB', 91 * 1024**8)])
    def test_size_limit_human_readable(self, local_remote_dir: Tuple[str, str], input: str,
                                       expected: int) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        mwriter = MDSWriter(out=local,
                            columns=columns,
                            compression=None,
                            hashes=None,
                            size_limit=input)
        jwriter = JSONWriter(out=local,
                             columns=columns,
                             compression=None,
                             hashes=None,
                             size_limit=input)
        xwriter = XSVWriter(out=local,
                            separator=',',
                            columns=columns,
                            compression=None,
                            hashes=None,
                            size_limit=input)
        assert (mwriter.size_limit == expected)
        assert (jwriter.size_limit == expected)
        assert (xwriter.size_limit == expected)

    def test_size_limit_negative_MDS(self, local_remote_dir: Tuple[str, str]) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        with pytest.raises(ValueError):
            _ = MDSWriter(out=local,
                          columns=columns,
                          compression=None,
                          hashes=None,
                          size_limit='-4GB')

    def test_size_limit_negative_JSON(self, local_remote_dir: Tuple[str, str]) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        with pytest.raises(ValueError):
            _ = JSONWriter(out=local,
                           columns=columns,
                           compression=None,
                           hashes=None,
                           size_limit='-4GB')

    def test_size_limit_negative_XSV(self, local_remote_dir: Tuple[str, str]) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        with pytest.raises(ValueError):
            _ = XSVWriter(out=local,
                          separator=',',
                          columns=columns,
                          compression=None,
                          hashes=None,
                          size_limit='-4GB')

    def test_size_limit_not_given(self, local_remote_dir: Tuple[str, str]) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        mwriter = MDSWriter(out=local, columns=columns, compression=None, hashes=None)
        jwriter = JSONWriter(out=local, columns=columns, compression=None, hashes=None)
        xwriter = XSVWriter(out=local,
                            separator=',',
                            columns=columns,
                            compression=None,
                            hashes=None)
        assert (mwriter.size_limit == 1 << 26)
        assert (jwriter.size_limit == 1 << 26)
        assert (xwriter.size_limit == 1 << 26)

    def test_size_limit_none(self, local_remote_dir: Tuple[str, str]) -> None:
        local, _ = local_remote_dir
        dataset = SequenceDataset(100)
        columns = dict(zip(dataset.column_names, dataset.column_encodings))
        mwriter = MDSWriter(out=local,
                            columns=columns,
                            compression=None,
                            hashes=None,
                            size_limit=None)
        jwriter = JSONWriter(out=local,
                             columns=columns,
                             compression=None,
                             hashes=None,
                             size_limit=None)
        xwriter = XSVWriter(out=local,
                            separator=',',
                            columns=columns,
                            compression=None,
                            hashes=None,
                            size_limit=None)
        assert (mwriter.size_limit is None)
        assert (jwriter.size_limit is None)
        assert (xwriter.size_limit is None)
