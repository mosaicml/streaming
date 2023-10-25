# Hashing

Streaming supports a variety of hash and checksum algorithms to verify data integrity.

We optionally hash shards while serializing a streaming dataset, saving the resulting hashes in the index, which is written last. After the dataset is finished being written, we may hash the index file itself, the results of which must be stored elsewhere. Hashing during writing is controlled by the Writer argument `hashes: Optional[List[str]] = None`. We generally weakly recommend writing streaming datasets with one cryptographic hash algorithm and one fast hash algorithm for offline dataset validation in the future.

Then, we optionally validate shard hashes upon download while reading a streaming dataset. Hashing during reading is controlled separately by the StreamingDataset argument `validate_hash: Optional[List[str]] = None`. We recommend reading streaming datasets for training purposes without validating hashes because of the extra cost in time and computation.

Available cryptographic hash functions:

| Hash     | Digest Bytes |
| -------- | ------------ |
| blake2b  | 64           |
| blake2s  | 32           |
| md5      | 16           |
| sha1     | 20           |
| sha224   | 28           |
| sha256   | 32           |
| sha384   | 48           |
| sha512   | 64           |
| sha3_224 | 28           |
| sha3_256 | 32           |
| sha3_384 | 48           |
| sha3_512 | 64           |

Available non-cryptographic hash functions:

| Hash     | Digest Bytes |
| -------- | ------------ |
| xxh32    | 4            |
| xxh64    | 8            |
| xxh128   | 16           |
| xxh3_64  | 8            |
| xxh3_128 | 16           |
