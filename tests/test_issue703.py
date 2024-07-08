
import numpy as np
from PIL import Image
from shutil import rmtree
from uuid import uuid4
from streaming import MDSWriter

# Local or remote directory path to store the output compressed files.
out_root = "dbfs:/Volumes/**name.schema/volume_name**/test"
out_root = "dbfs:/Volumes/main/mosaic_hackathon/managed-volume/test_issue703"
#out_root = '/dbfs/FileStore/test'
#os.environ['DATABRICKS_HOST'] = 'https://**address**.azuredatabricks.net'
#os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(..... )
# A dictionary of input fields to an Encoder/Decoder type
columns = {
    'uuid': 'str',
    'img': 'jpeg',
    'clf': 'int'
}

# Compression algorithm name
compression = 'zstd'

# Generate random images and classes
samples = [
    {
        'uuid': str(uuid4()),
        'img': Image.fromarray(np.random.randint(0, 256, (32, 48, 3), np.uint8)),
        'clf': np.random.randint(10),
    }
    for _ in range(1000)
]

# Use `MDSWriter` to iterate through the input data and write to a collection of `.mds` files.
with MDSWriter(out=out_root, columns=columns, compression=compression) as out:
    for sample in samples:
        out.write(sample)
