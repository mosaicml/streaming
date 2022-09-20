# Training CIFAR-10
In this example, we will demonstrate how to compress the CIFAR-10 dataset using {class}`streaming.MDSWriter` and use {class}`streaming.Dataset` to train a CIFAR-10 classifier using ResNet-9 architecture.

Below is the step by step tutorial. 

## Step 1: Compress the CIFAR-10 dataset
Use the existing [cifar10.py](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/cifar10.py) script which uses {class}`streaming.MDSWriter` to compress the raw data.

```bash
# Download the script locally
$ wget https://raw.githubusercontent.com/mosaicml/streaming/main/streaming/vision/convert/cifar10.py

$ python cifar10.py --in_root <local_directory_to_download_raw_CIFAR10_data> --out_root <local_directory_to_store_compressed_files>
```

## Step 2: Upload the compressed files to cloud
Upload the `--out_root` directory to cloud blob storage such as AWS S3 using [AWS CLI](https://aws.amazon.com/cli/).

```bash
$ aws s3 cp <out_root directory> s3://mybucket/cifar10 --recursive
```

## Step 3: Import packages for model training
```python
import time

import composer
import torch
import torch.utils.data
from torchvision import transforms
```

## Step 4: Initialize transforms
```python
# CIFAR10 mean and standard deviation for normalization.
CIFAR10_MEAN = 0.4914, 0.4822, 0.4465
CIFAR10_STD = 0.247, 0.243, 0.261

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
```
## Step 5: Create a streaming Dataset
Here, we have used {class}`streaming.vision.CIFAR10` which is a superclass of {class}`streaming.Dataset` for better abstraction.

```python
batch_size = 1024   # Batch size to use
local_dir = '/local/directory/to/cache/streaming/data'
remote_dir = 's3://mybucket/cifar10'

from streaming.vision import CIFAR10
train_dataset = CIFAR10(local=local_dir,
                        remote=remote_dir,
                        split='train',
                        shuffle=True,
                        transform=train_transform,
                        batch_size=batch_size)

val_dataset = CIFAR10(local=local_dir,
                      remote=remote_dir,
                      split='val',
                      shuffle=False,
                      transform=train_transform,
                      batch_size=batch_size)
```
## Step 6: Create a PyTorch DataLoader
Create a PyTorch {class}`torch.utils.data.DataLoader` with {class}`streaming.vision.CIFAR10` dataset.

```python
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               prefetch_factor=2,
                                               num_workers=8)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             prefetch_factor=2,
                                             num_workers=8)
```

## Step 7: Setup model architecture and optimization parameters
For the model, we use a custom ResNet-9 architecture

```python
from composer import models

model = models.composer_resnet_cifar(model_name='resnet_9', num_classes=10)

optimizer = composer.optim.DecoupledSGDW(
    model.parameters(),  # Model parameters to update
    lr=0.05,  # Peak learning rate
    momentum=0.9,
    weight_decay=
    1e-4  # If this looks large, it's because its not scaled by the LR as in non-decoupled weight decay
)

lr_scheduler = composer.optim.LinearWithWarmupScheduler(
    t_warmup='1ep',  # Warm up over 1 epoch
    alpha_i=1.0,  # Flat LR schedule achieved by having alpha_i == alpha_f
    alpha_f=1.0)

train_epochs = '10ep'  # Train for 10 epochs
device = 'gpu' if torch.cuda.is_available() else 'cpu'  # select the device
```

## Step 8: Train and evaluate the model
Trains the model for 10 epochs with evaluation at every epoch.

```python
start_time = time.perf_counter()
trainer.fit()
end_time = time.perf_counter()
print(f'It took {end_time - start_time:0.4f} seconds to train')
```

## Wrapping up
In this example, we show how to create a sharded dataset and use the streaming dataset to train a CIFAR-10 classifier model.