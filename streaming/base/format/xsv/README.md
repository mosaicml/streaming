Example:

    {
      "column_encodings": [
        "int",
        "str"
      ],
      "column_names": [
        "number",
        "words"
      ],
      "compression": "zstd:7",
      "format": "csv",
      "hashes": [
        "sha1",
        "xxh3_64"
      ],
      "newline": "\n",
      "raw_data": {
        "basename": "shard.00000.csv",
        "bytes": 1048523,
        "hashes": {
          "sha1": "39f6ea99d882d3652e34fe5bd4682454664efeda",
          "xxh3_64": "ea1572efa0207ff6"
        }
      },
      "raw_meta": {
        "basename": "shard.00000.csv.meta",
        "bytes": 77486,
        "hashes": {
          "sha1": "8874e88494214b45f807098dab9e55d59b6c4aec",
          "xxh3_64": "3b1837601382af2c"
        }
      },
      "samples": 19315,
      "separator": ",",
      "size_limit": 1048576,
      "version": 2,
      "zip_data": {
        "basename": "shard.00000.csv.zstd",
        "bytes": 197040,
        "hashes": {
          "sha1": "021d288a317ae0ecacba8a1b985ee107f966710d",
          "xxh3_64": "5daa4fd69d3578e4"
        }
      },
      "zip_meta": {
        "basename": "shard.00000.csv.meta.zstd",
        "bytes": 60981,
        "hashes": {
          "sha1": "f2a35f65279fbc45e8996fa599b25290608990b2",
          "xxh3_64": "7c38dee2b3980deb"
        }
      }
    }
