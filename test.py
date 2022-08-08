from shutil import rmtree

import numpy as np

from streaming import CSVWriter, Dataset, JSONWriter, MDSWriter, TSVWriter

ones = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen ' +
        'fifteen sixteen seventeen eighteen nineteen').split()

tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def say(i):
    if i < 0:
        return ['negative'] + say(-i)
    elif i <= 19:
        return [ones[i]]
    elif i < 100:
        return [tens[i // 10 - 2]] + ([ones[i % 10]] if i % 10 else [])
    elif i < 1_000:
        return [ones[i // 100], 'hundred'] + (say(i % 100) if i % 100 else [])
    elif i < 1_000_000:
        return say(i // 1_000) + ['thousand'] + (say(i % 1_000) if i % 1_000 else [])
    elif i < 1_000_000_000:
        return say(i // 1_000_000) + ['million'] + (say(i % 1_000_000) if i % 1_000_000 else [])
    else:
        assert False


def get_number():
    sign = (np.random.random() < 0.8) * 2 - 1
    mag = 10 ** np.random.uniform(1, 4) - 10
    return sign * int(mag ** 2)


def get_dataset(num_samples):
    samples = []
    for i in range(num_samples):
        number = get_number()
        words = ' '.join(say(number))
        sample = {
            'number': number,
            'words': words
        }
        samples.append(sample)
    return samples


def main():
    samples = get_dataset(50_000)

    columns = {
        'number': 'int',
        'words': 'str',
    }
    compression = 'zstd:7'
    hashes = 'sha1', 'xxh3_64'
    size_limit = 1 << 16

    with MDSWriter('/tmp/mds', columns, compression, hashes, size_limit) as out:
        for x in samples:
            out.write(x)

    with CSVWriter('/tmp/csv', columns, compression, hashes, size_limit) as out:
        for x in samples:
            out.write(x)

    with TSVWriter('/tmp/tsv', columns, compression, hashes, size_limit) as out:
        for x in samples:
            out.write(x)

    with JSONWriter('/tmp/json', columns, compression, hashes, size_limit) as out:
        for x in samples:
            out.write(x)

    for dirname in ['/tmp/mds', '/tmp/csv', '/tmp/tsv', '/tmp/json']:
        dataset = Dataset(dirname, shuffle=False)
        for gold, test in zip(samples, dataset):
            assert gold == test

    for dirname in ['/tmp/mds', '/tmp/csv', '/tmp/tsv', '/tmp/json']:
        rmtree(dirname)


main()
