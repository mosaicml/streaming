from streaming.base.dataset import StreamingDataset
from streaming.base.world import World

import torch
import os

__all__ = ['MegatronStreamingDataset']


class PerNodeWorld(World):
    @classmethod
    def detect(cls):
        from streaming.base.megatron_dataset_utils import get_dataset_builder_ranks_by_node, get_parallel_rank_info
        # global _DATASET_BUILDER_RANKS_BY_NODE
        all_ranks_per_node = get_dataset_builder_ranks_by_node()
        parallel_rank_info = get_parallel_rank_info()
        node = torch.distributed.get_rank() // int(os.environ["LOCAL_WORLD_SIZE"])
        rank = all_ranks_per_node[node][parallel_rank_info]
        ranks_for_node = len(all_ranks_per_node[node])
        num_nodes = 1 
        worker_of_rank, workers_per_rank = cls._get_worker_info()
        worker = rank * workers_per_rank + worker_of_rank
        return cls(num_nodes, ranks_for_node, workers_per_rank, worker)
    
    def replicate(self, replication):
        raise NotImplementedError(f'PerNodeWorld does not support replicate()')
    
class DPWorld(World):
    @classmethod
    def detect(cls):
        from megatron.core import mpu
        rank = mpu.get_data_parallel_rank()
        ranks_per_node = mpu.get_data_parallel_world_size()
        num_nodes = 1
        worker_of_rank, workers_per_rank = cls._get_worker_info()
        worker = rank * workers_per_rank + worker_of_rank
        return cls(num_nodes, ranks_per_node, workers_per_rank, worker)
    
    def replicate(self, replication):
        raise NotImplementedError(f'DPWorld does not support replicate()')
    

class MegatronStreamingDataset(StreamingDataset):
    """A dataset class compatible with Megatron-LM's data loading. This dataset can be used in place of Megatron's BlendedMegatronDataset

    This class extends the base StreamingDataset class to ensure compatibility with Megatron-LM's distributed data loading policies for N-D parallelisms.
    """

    def __init__(self, *args, **kwargs):
        replication = kwargs.pop('replication', None)
        assert replication is None, 'MegatronStreamingDataset does not support replication. Please remove the `replication` argument.'
        super().__init__(*args, **kwargs)
        print(f'Initialized MegatronStreamingDataset\n', flush=True)

    def _create_unique_rank_world(self) -> World:
        return PerNodeWorld.detect()
    
    def _create_parallel_rank_world(self) -> World:
        return DPWorld.detect()