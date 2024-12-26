from paddle.io import DataLoader
from streaming import StreamingDataset
import paddle.distributed as dist
from pudb.remote import set_trace

dist.init_parallel_env()

# Remote path where full dataset is persistently stored
# remote = 's3://path-to-dataset'
remote = 'dirname'

# Local working dir where dataset is cached during operation
local = 'cache'

# Create streaming dataset
dataset = StreamingDataset(local=local, remote=remote, shuffle=False, num_canonical_nodes=1, batch_size=1)

# Let's see what is in sample #1337...
sample = dataset[2]
img = sample['image']
cls = sample['class']

# Create PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=1)

for idx, i in enumerate(dataloader):
    # set_trace()
    # if dist.get_rank() == 3:
        # import pdb;pdb.set_trace()
    print(f"Rank: {dist.get_rank()}, data: {i}, idx: {idx}")