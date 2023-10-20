from streaming import StreamingDataset, StreamingDataLoader

from streaming import StreamingDataset
from streaming import StreamingDataLoader

local_path = "/Users/saaketh.narayan/Desktop/temp/datasets/fake_stream_1"

dataset = StreamingDataset(local=local_path, partition_algo="relaxed")
dataloader = StreamingDataLoader(dataset, batch_size=1)

state_dict = None
for i, batch in enumerate(dataloader):
    print(i, batch)
    if i == 4:
        state_dict = dataloader.state_dict()
    if i == 5:
        break

print(state_dict)
state_dict['initial_physical_nodes'] = 2

dataset2 = StreamingDataset(local=local_path, partition_algo="relaxed", batch_size=2)
dataloader2 = StreamingDataLoader(dataset2, batch_size=2)
dataloader2.load_state_dict(state_dict)

for i, batch in enumerate(dataloader2):
    print(i, batch)
    if i == 1:
        break