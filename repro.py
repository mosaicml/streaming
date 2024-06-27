from streaming.base import StreamingDataset
from torch.utils.data import DataLoader
import streaming
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
#from torchrec.test_utils import get_free_port
import os
import socket
import multiprocessing

def get_free_tcp_port() -> int:
    """Get free socket port to use as MASTER_PORT."""
    # from https://www.programcreek.com/python/?CodeExample=get+free+port
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def fun(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)


    #streaming.base.util.clean_stale_shared_memory()
    dataset = StreamingDataset(
        remote="s3://my-bucket/my-copy-arxiv",
        local="./local/",
        keep_zip=False,
        batch_size=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
    )

    for d in dataloader:
        print(d)
    print("haha")

def main():
    processes = []
    world_size = 8
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
    os.environ["MASTER_PORT"] = str(get_free_tcp_port())
    os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    ctx = multiprocessing.get_context("spawn")  
    for rank in range(world_size):
        p = ctx.Process(
            target=fun,
            args=(
                rank,
                world_size,
            )
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    


if __name__ == "__main__":
    main()
    
