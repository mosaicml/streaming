Example:

'''
    {
      "column_encodings": [
        "int",
        "str"
      ],
      "column_names": [
        "number",
        "words"
      ],
      "column_sizes": [
        8,
        null
      ],
      "compression": "zstd:7",
      "format": "mds",
      "hashes": [
        "sha1",
        "xxh3_64"
      ],
      "raw_data": {
        "basename": "shard.00000.mds",
        "bytes": 1048544,
        "hashes": {
          "sha1": "8d0634d3836110b00ae435bbbabd1739f3bbeeac",
          "xxh3_64": "2c54988514bca807"
        }
      },
      "samples": 16621,
      "size_limit": 1048576,
      "version": 2,
      "zip_data": {
        "basename": "shard.00000.mds.zstd",
        "bytes": 228795,
        "hashes": {
          "sha1": "2fb5ece19aabc91c2d6d6c126d614ab291abe24a",
          "xxh3_64": "fe6e78d7c73d9e79"
        }
      }
    }
'''
