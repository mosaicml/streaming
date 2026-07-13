import json
import glob
import os
import shutil

from typing import Callable, Optional, Dict, List, Iterable, Any
from argparse import Namespace
import torch
from torch.utils.data._utils.collate import default_collate
from megatron.training.global_vars import _GLOBAL_ARGS, _ensure_var_is_not_initialized, set_args

def set_global_variables(args: Namespace, build_tokenizer: bool = False):
    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    set_args(args)

def _compile_dependencies():
    return

def validate_args(args: Namespace, defaults: Dict ={}):
    """Validate arguments for this demo."""

    if args.pipeline_model_parallel_size != 1:
        raise ValueError("This demo only supports pipeline_model_parallel_size=1")

    assert args.world_size %  args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size *args.expert_model_parallel_size == 0, \
        "The world size must divide the product of tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size and expert_model_parallel_size"

    args.data_parallel_size = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size *args.expert_model_parallel_size)
    args.virtual_pipeline_model_parallel_size = None

    print_rank_0(f"Data parallel size: {args.data_parallel_size}")

    print_rank_0(f"Tensor parallel size: {args.tensor_model_parallel_size}")

    print_rank_0(f"Context parallel size: {args.context_parallel_size}")

    print_rank_0(f"Expert parallel size: {args.expert_model_parallel_size}")

    print_rank_0(f"micro batch size: {args.micro_batch_size}")




from megatron.training import initialize

# override validate_args with the simpler check we need for this demo
initialize.validate_args = validate_args

# override to do nothing for this demo
initialize.set_global_variables = set_global_variables

initialize._compile_dependencies = _compile_dependencies

from megatron.training import get_args
from megatron.core import mpu
from megatron.training.initialize import initialize_megatron

from megatron.training.utils import print_rank_0

from streaming.base import MegatronStreamingDataset, MegatronStreamingDataLoader
from streaming.base.megatron_dataset_utils import build_dataset_builder_ranks_by_node, get_parallel_rank_info
from tests.common.utils import convert_to_mds

all_batch_info: Optional[Dict[str, List[int]]] = None
save_path = '/tmp/streaming_dataset_demo'


def init_all_batch_info():
    from dataclasses import asdict

    global all_batch_info
    world = get_parallel_rank_info()

    all_batch_info = asdict(world)

    all_batch_info['idxs'] = []

def load_all_batch_info(load_dir: str):
    node = torch.distributed.get_rank() // int(os.environ["LOCAL_WORLD_SIZE"])
    rank = torch.distributed.get_rank()
    

    with open(os.path.join(load_dir, f'node_{node}_rank_{rank}.json'), 'r') as f:
        data = json.load(f)
    
    global all_batch_info
    all_batch_info = data

def update_all_batch_info(batch: Dict[str, Any]):
    global all_batch_info
    if all_batch_info is None:
        init_all_batch_info()
    all_batch_info['idxs'].extend(batch['idx'].cpu().numpy().tolist())

def save_batch_info(save_dir: str):
    global all_batch_info
    rank = torch.distributed.get_rank()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    node = rank // local_world_size

    with open(os.path.join(save_dir, f'node_{node}_rank_{rank}.json'), 'w') as f:
        json.dump(all_batch_info, f)

def validate_batch_info(save_dir: str, expected_num_samples: int):
    all_idxs = []

    idxs_per_dp_rank = {}

    for filepath in glob.glob(os.path.join(save_dir, f'node_*_rank_*.json')):
        with open(filepath, 'r') as f:
            data = json.load(f)
            idxs_per_dp_rank.setdefault(data['data_parallel_rank'], []).append(data['idxs'])

    # Check that all dp_ranks see the same indices

    for dp_rank, idxs_list in idxs_per_dp_rank.items():
        for i in range(len(idxs_list)):
            for j in range(i + 1, len(idxs_list)):
                if idxs_list[i] != idxs_list[j]:
                    print(f'Error: Data parallel rank {dp_rank} has inconsistent indices across its ranks.\n', flush=True)
                    return
        all_idxs.extend(idxs_list[0])
        
                
    print('All data parallel ranks have consistent indices across their ranks.\n', flush=True)


    # Check for missing or duplicate indices
    all_idxs_set = set(all_idxs)
    if len(all_idxs) != expected_num_samples:
        print(f'Error: Expected {expected_num_samples} samples, but got {len(all_idxs)} samples.\n', flush=True)
    elif len(all_idxs_set) != expected_num_samples:
        print(f'Error: Found duplicate indices in the collected samples.\n', flush=True)
    else:
        print('All samples unique and all the data is streamed correctly\n', flush=True)


def is_dataset_built_on_rank():
    """We define a simple policy here that only ranks with tp_rank == 0 will build the dataset. 
    The corresponding Megatron function considers vp_stage and mtp_on_this_rank but for simplicity we ignore these in this demo.
    """
    return mpu.get_tensor_model_parallel_rank() == 0

def _get_iterator(dataloader: Optional[MegatronStreamingDataLoader]) -> Optional[Iterable]:
    if dataloader is None:
        return None
    else:
        return iter(dataloader)
    

def build_train_valid_test_datasets(build_train_valid_test_datasets_provider: Callable):
    return build_train_valid_test_datasets_provider()

        
def build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider: Callable):
    args = get_args()

    # We define a policy here that only ranks with tp_rank == 0 will build the dataset
    if is_dataset_built_on_rank():
        # Build datasets.
        train_ds, _, _ = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider)
        return MegatronStreamingDataLoader(
            train_ds,
            batch_size=args.micro_batch_size,
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=True,
            collate_fn=default_collate
        ), None, None
    else:
        return None, None, None
    
def train_valid_test_datasets_provider():
    remote_dir, local_dir = os.path.join(save_path, 'remote'), os.path.join(save_path, 'local')

    args = get_args()

    dataset = MegatronStreamingDataset(
        local=local_dir,
        remote=remote_dir,
        shuffle=True,
        num_canonical_nodes=1,
        batch_size=args.micro_batch_size,
    )

    return dataset, None, None

    
def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider: Callable):
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider)
    train_data_iterator = _get_iterator(train_dataloader)
    valid_data_iterator = _get_iterator(valid_dataloader)
    test_data_iterator = _get_iterator(test_dataloader)
    return train_data_iterator, valid_data_iterator, test_data_iterator


def iterate_dataset(
    train_valid_test_dataset_provider: Callable,
    get_batch: Callable,
    is_dataset_built_on_rank_func: Callable,
    args_defaults: Dict[str, Any]={},

):
    """Data iteration loop.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) Iterate train data.

    Args:
        train_valid_test_dataset_provider: a function that returns the
            train/valid/test dataset and returns `train, valid, test` datasets.
        is_dataset_built_on_rank_func: a function that returns True if dataset is built on a rank and False otherwise
    """

    assert is_dataset_built_on_rank_func is not None, \
        "is_dataset_built_on_rank_func must be provided to iterate_data."
    
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        args_defaults=args_defaults,
    )

    global save_path

    remote_dir = os.path.join(save_path, 'remote')

    if torch.distributed.get_rank() == 0:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            print(f"Directory '{save_path}' has been removed.")
        else:
            print(f"Directory '{save_path}' does not exist or is not a directory.")
            
        os.makedirs(save_path)
        
        convert_to_mds(out_root=remote_dir,
                    dataset_name='sequencedatasetint',
                    num_samples=1024,
                    size_limit=1 << 8)
    
    torch.distributed.barrier()

    args = get_args()


    print_rank_0(f"Starting job with local world size: {os.environ['LOCAL_WORLD_SIZE']}")

    # Essential to call this function, which initializes all the info that StreamingDataset needs to know about the Megatron parallel environment (e.g. which ranks build the dataset, how to share data across ranks, etc.)
    build_dataset_builder_ranks_by_node(is_dataset_built_on_rank_func)


    iter = 0

    train_data_iterator, _, _  = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)

    print_freq = 10

    while iter < args.train_iters:
        batch = get_batch(train_data_iterator)
        if batch is None:
            break
        iter += 1
        if iter % print_freq == 0:
            print(f'Processing batch {iter} on rank {torch.distributed.get_rank()} with keys: {list(batch.keys())}\n', flush=True)

        update_all_batch_info(batch)

    print(f'Finished iterating through data on rank {torch.distributed.get_rank()}. Saving batch info...', flush=True)
    save_batch_info(save_path)

    torch.distributed.barrier()
    validate_batch_info(save_path, expected_num_samples=args.micro_batch_size * mpu.get_data_parallel_world_size() * iter)

    torch.distributed.destroy_process_group()

def get_batch_on_this_cp_rank(batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """In a real implementation with token sequences, we would slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """
    return batch

def get_batch_on_this_tp_rank(data_iterator: Iterable) -> Optional[Dict[str, Any]]:
    """Broadcast from tp_rank=0 to other ranks in the same tp group"""
    args = get_args()

    device = torch.cuda.current_device()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_src = mpu.get_tensor_model_parallel_src_rank()

    mbs = args.micro_batch_size

    # ------------------------------------------------------------------
    # 1. Pre-allocate tensors on ALL TP ranks (fixed shapes)
    # ------------------------------------------------------------------
    batch = {
        "idx": torch.empty((mbs,), dtype=torch.int64, device=device),
    }

    # Scalar "is valid batch" flag
    flag = torch.zeros((), dtype=torch.int8, device=device)

    # ------------------------------------------------------------------
    # 2. Only TP src rank touches the iterator
    # ------------------------------------------------------------------
    if tp_rank == 0:
        try:
            src_batch = next(data_iterator)
            flag.fill_(1)
            actual_bs = src_batch["idx"].shape[0]

            # ------------------------------------------------------------------------------------
            # 2a. Pad last batch if needed. This can be avoided with drop_last=True in DataLoader
            # ------------------------------------------------------------------------------------
            if actual_bs < mbs:
                pad = mbs - actual_bs
                for k, v in src_batch.items():
                    pad_shape = (pad, *v.shape[1:])
                    pad_tensor = torch.zeros(
                        pad_shape, dtype=v.dtype, device=v.device
                    )
                    src_batch[k] = torch.cat([v, pad_tensor], dim=0)

            # ----------------------------------------------------------
            # 2b. Copy into fixed-shape buffers
            # ----------------------------------------------------------
            for k in batch:
                batch[k].copy_(src_batch[k], non_blocking=True)

        except StopIteration:
            flag.fill_(0)

    # ------------------------------------------------------------------
    # 3. Broadcast flag (ALWAYS)
    # ------------------------------------------------------------------
    torch.distributed.broadcast(flag, tp_src, group=tp_group)

    # ------------------------------------------------------------------
    # 4. Broadcast tensors (ALWAYS, even if flag == 0)
    # In PP/VP the broadcast depends on PP/VP stage but we do not worry about such oprimizations here
    # ------------------------------------------------------------------
    for v in batch.values():
        torch.distributed.broadcast(v, tp_src, group=tp_group)

    # ------------------------------------------------------------------
    # 5. Consume flag
    # ------------------------------------------------------------------
    if flag.item() == 0:
        return None

    return batch


def get_batch(data_iterator: Optional[Iterable]) -> Optional[Dict[str, Any]]:
    """Generate a batch."""
    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch

if __name__ == "__main__":
    iterate_dataset(train_valid_test_dataset_provider=train_valid_test_datasets_provider, 
                    get_batch=get_batch, 
                    is_dataset_built_on_rank_func=is_dataset_built_on_rank,
                )