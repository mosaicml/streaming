# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from simulation.core.sim_dataset import SimulationDataset
from simulation.core.sim_time import Time
from simulation.core.utils import get_batches_epochs


@pytest.mark.parametrize(('max_batches', 'expected_epochs'), [(94_000, 2), (100_000, 3)])
def test_batch_duration_spanning_epochs(max_batches: int, expected_epochs: int) -> None:
    dataset = Mock(spec=SimulationDataset)
    dataset.get_num_batches.return_value = 47_000

    result = get_batches_epochs(dataset, Time.from_batch(max_batches))

    assert result == (47_000, expected_epochs, max_batches)
