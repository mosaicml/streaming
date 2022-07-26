import json

from streaming.base.compression import get_compressions
from streaming.base.hashing import get_hashes
from streaming.base.mds.encodings import get_encodings


def main():
    compressions = sorted(get_compressions())
    hashes = sorted(get_hashes())
    encodings = sorted(get_encodings())
    obj = {
        'compressions': compressions,
        'hashes': hashes,
        'encodings': encodings,
    }
    print(json.dumps(obj, indent=4, sort_keys=True))
    

if __name__ == '__main__':
    main()
