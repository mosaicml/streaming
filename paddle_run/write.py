import numpy as np
from streaming import MDSWriter

# Directory in which to store the compressed output files
data_dir = 'dirname'

# A dictionary mapping input fields to their data types
columns = {
    'image': 'ndarray',
    'class': 'int'
}

# Shard compression, if any
compression = 'zstd'

# Save the samples as shards using MDSWriter
with MDSWriter(out=data_dir, columns=columns, compression=compression) as out:
    for i in range(8):
        sample = {
            'image': np.random.randint(i+1, i+2, (1, 1)),
            'class': np.random.randint(i+1, i+2),
        }
        print(sample)
        out.write(sample)
