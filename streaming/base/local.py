import json
import os
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from .format import reader_from_json
from .index import Index


class LocalDataset(Dataset):

    def __init__(self, dirname: str, split: Optional[str] = None):
        split = split or ''

        self.dirname = dirname
        self.split = split

        filename = os.path.join(dirname, split, 'index.json')
        obj = json.load(open(filename))
        assert obj['version'] == 2

        self.shards = []
        for info in obj['shards']:
            shard = reader_from_json(dirname, split, info)
            self.shards.append(shard)

        shard_sizes = list(map(lambda x: x.samples, self.shards))
        self.index = Index(shard_sizes)

    def __len__(self) -> int:
        return self.index.total_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx, idx_in_shard = self.index.find_sample(idx)
        shard = self.shards[shard_idx]
        return shard[idx_in_shard]
