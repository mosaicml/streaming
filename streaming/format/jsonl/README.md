Example:

```json
    {
      "columns": {
        "number": "int",
        "words": "str"
      },
      "compression": "zstd:7",
      "format": "jsonl",
      "hashes": [
        "sha1",
        "xxh3_64"
      ],
      "newline": "\n",
      "raw_data": {
        "basename": "shard.00000.jsonl",
        "bytes": 1048546,
        "hashes": {
          "sha1": "bfb6509ba6f041726943ce529b36a1cb74e33957",
          "xxh3_64": "0eb102a981b299eb"
        }
      },
      "raw_meta": {
        "basename": "shard.00000.jsonl.meta",
        "bytes": 53590,
        "hashes": {
          "sha1": "15ae80e002fe625b0b18f1a45058532ee867fa9b",
          "xxh3_64": "7b113f574a422ac1"
        }
      },
      "samples": 13352,
      "size_limit": 1048576,
      "version": 2,
      "zip_data": {
        "basename": "shard.00000.jsonl.zstd",
        "bytes": 149268,
        "hashes": {
          "sha1": "7d45c600a71066ca8d43dbbaa2ffce50a91b735e",
          "xxh3_64": "3d338d4826d4b5ac"
        }
      },
      "zip_meta": {
        "basename": "shard.00000.jsonl.meta.zstd",
        "bytes": 42180,
        "hashes": {
          "sha1": "f64477cca5d27fc3a0301eeb4452ef7310cbf670",
          "xxh3_64": "6e2b364f4e78670d"
        }
      }
    }
```
