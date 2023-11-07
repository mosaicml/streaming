# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from streaming.base import StreamingDataset
from tests.common.utils import convert_to_mds
import pytest
from typing import Tuple
from streaming.base.util import clean_stale_shared_memory

@pytest.mark.usefixtures('local_remote_dir')
def test_new_defaults_warning(local_remote_dir: Tuple[str,str],):
    local, remote = local_remote_dir
    convert_to_mds(out_root=remote,
                   dataset_name='sequencedataset',
                   num_samples=300,
                   size_limit=1 << 8)

    with pytest.warns(UserWarning, match=f'Because `predownload` was not specified,*'):
        # Build a StreamingDataset with new defaults. Should warn about the new defaults changes.
        _ = StreamingDataset(local=local, remote=remote, shuffle=True)
    
    clean_stale_shared_memory()

    with pytest.warns(UserWarning, match=f'Because `shuffle_block_size` was not specified,*'):
        # Build a StreamingDataset with new defaults. Should warn about the new defaults changes.
        for _ in StreamingDataset(local=local, remote=remote, shuffle=True):
            pass

    clean_stale_shared_memory()

    with pytest.warns(UserWarning, match=f'Because `num_canonical_nodes` was not specified,*'):
        # Build a StreamingDataset with new defaults. Should warn about the new defaults changes.
        for _ in StreamingDataset(local=local, remote=remote, shuffle=True):
                pass
        
    clean_stale_shared_memory()


