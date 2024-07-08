from streaming import StreamingDataset

# Create streaming dataset
dataset = StreamingDataset(remote="hf://datasets/orionweller/wikipedia_mds/", shuffle=False, split=None, batch_size=1)

# Let's see what's in it
for sample in dataset:
    text = sample['text']
    id = sample['id']
    print(f"Text: {text}")
    print(f"ID: {id}")
    break
