import numpy as np

class RandomClassificationDataset:
    """Classification dataset drawn from a normal distribution.

    Args:
        shape: data sample dimensions (default: (10,))
        size: number of samples (default: 10000)
        num_classes: number of classes (default: 2)
    """

    def __init__(self, shape=(10,), size=10000, num_classes=2):
        self.size = size
        self.x = np.random.randn(size, *shape)
        self.y = np.random.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index], int(self.y[index])
    
output_dir = 'datasets/my_data'
columns = {'x': 'pkl', 'y': 'int'}
compression = 'zstd:7'
hashes = ['sha1']
limit = '10kb'

def each(samples):
    """Generator over each raw dataset sample.

    Args:
        samples: Raw samples as tuples of (feature, label).

    Yields:
        Each sample, as a dict.
    """
    for x, y in samples:
        yield {
            'x': x,
            'y': y,
        }

from streaming.base import MDSWriter

dataset = RandomClassificationDataset()
with MDSWriter(out=output_dir, columns=columns, compression=compression, hashes=hashes, size_limit=limit) as out:
    for sample in each(dataset):
        out.write(sample)