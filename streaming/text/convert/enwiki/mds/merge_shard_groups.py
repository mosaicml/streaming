from glob import glob
import json
import os
import shutil

in_root = '/tmp/mds-enwiki/train/'
subdir_pattern = os.path.join(in_root, 'group-*')
out_root = '/dataset/mds-enwiki/train/'
os.makedirs(out_root)
subdirs = sorted(glob(subdir_pattern))
offset = 0
infos = []
for subdir in subdirs:
    shards_this_group = len(os.listdir(subdir)) - 1

    # Move shard files.
    for shard in range(shards_this_group):
        old_filename = f'{subdir}/shard.{shard:05d}.mds.zstd'
        new_filename = f'{out_root}/shard.{offset + shard:05d}.mds.zstd'
        os.rename(old_filename, new_filename)

    # Collect shard infos.
    index_filename = f'{subdir}/index.json'
    obj = json.load(open(index_filename))
    infos += obj['shards']

    # Update offset.
    offset += shards_this_group

# Update the indices of the collected shard infos to be global.
for shard, info in enumerate(infos):
    info['raw_data']['basename'] = f'shard.{shard:05d}.mds'
    info['zip_data']['basename'] = f'shard.{shard:05d}.mds.zstd'

# Create new index.
obj = {
    'version': 2,
    'shards': infos,
}
index_filename = f'{out_root}/index.json'
with open(index_filename, 'w') as out:
    json.dump(obj, out)

# Remove leftover old index files.
shutil.rmtree(in_root)
