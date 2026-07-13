from dataclasses import dataclass
import os
from typing import Callable
import torch
import torch.distributed as dist
from megatron.core import mpu

_DATASET_BUILDER_RANKS_BY_NODE = None
_DATASET_BUILDING_GROUP = None

@dataclass(frozen=True)
class ParallelRankInfo:
    rank: int   
    tensor_model_parallel_rank: int
    context_parallel_rank: int
    pipeline_model_parallel_rank: int
    expert_parallel_rank: int
    data_parallel_rank: int

def get_parallel_rank_info():
    return ParallelRankInfo(
        rank=torch.distributed.get_rank(),
        tensor_model_parallel_rank=mpu.get_tensor_model_parallel_rank(),
        context_parallel_rank=mpu.get_context_parallel_rank(),
        pipeline_model_parallel_rank=mpu.get_pipeline_model_parallel_rank(),
        expert_parallel_rank=mpu.get_expert_model_parallel_rank(),
        data_parallel_rank=mpu.get_data_parallel_rank(),
    )

def build_dataset_builder_ranks_by_node(is_dataset_built_on_rank_func: Callable):
    """
    Returns:
        dict[int, list[tuple]]:
            {
              node_id: [
                (global_rank, tp_rank, cp_rank, pp_rank, ep_rank, dp_rank),
                ...
              ],
              ...
            }
    Only includes ranks for which is_dataset_built_on_rank_func() == True.
    """

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    node_id = rank // local_world_size

    tp_rank = mpu.get_tensor_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    ep_rank = mpu.get_expert_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()

    builds_dataset = is_dataset_built_on_rank_func()

    local_payload = torch.tensor(
        [
            node_id, 
            rank,
            tp_rank,
            cp_rank,
            pp_rank,
            ep_rank,
            dp_rank,
            int(builds_dataset),
        ],
        device="cuda",
        dtype=torch.int64,
    )

    gathered = [
        torch.empty_like(local_payload) for _ in range(world_size)
    ]
    dist.all_gather(gathered, local_payload)

    dataset_building_group = []

    result = {}

    for entry in gathered:
        (
            entry_node_id,
            entry_rank,
            entry_tp,
            entry_cp,
            entry_pp,
            entry_ep,
            entry_dp,
            entry_builds,
        ) = entry.tolist()

        if not entry_builds:
            continue
    
        dataset_building_group.append(entry_rank)

        result.setdefault(entry_node_id, []).append(
            (entry_rank, entry_tp, entry_cp, entry_pp, entry_ep, entry_dp)
        )

    p_results = {}
    for node_id, rank_tuples in result.items():
        p_results[node_id] = {
            ParallelRankInfo(
                rank=r,
                tensor_model_parallel_rank=tp,
                context_parallel_rank=cp,
                pipeline_model_parallel_rank=pp,
                expert_parallel_rank=ep,
                data_parallel_rank=dp,
            ) : i
            for i, (r, tp, cp, pp, ep, dp) in enumerate(rank_tuples)
        }

    global _DATASET_BUILDER_RANKS_BY_NODE
    _DATASET_BUILDER_RANKS_BY_NODE = p_results

    global _DATASET_BUILDING_GROUP
    _DATASET_BUILDING_GROUP = dist.new_group(ranks=dataset_building_group)

    print(f'dataset_building_ranks: {dataset_building_group}\n', flush=True)



    print(f"Built dataset builder ranks by node: {_DATASET_BUILDER_RANKS_BY_NODE}\n", flush=True)

def get_dataset_builder_ranks_by_node():
    if _DATASET_BUILDER_RANKS_BY_NODE is None:
        raise ValueError("Dataset builder ranks by node not built yet. Call build_dataset_builder_ranks_by_node() first.")
    
    return _DATASET_BUILDER_RANKS_BY_NODE

def get_dataset_building_group():
    if _DATASET_BUILDING_GROUP is None:
        raise ValueError("Dataset building group not built yet. Call build_dataset_builder_ranks_by_node() first.")
    
    return _DATASET_BUILDING_GROUP
