from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
import os
import tensorflow.compat.v1 as tf


def parse_args():
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, required=True)
    return args.parse_args()


def get_num_samples(filename):
    dataset = tf.data.TFRecordDataset(filename)
    count = 0
    for _ in dataset:
        count += 1
    return count


def main(args):
    pattern = os.path.join(args.dataset, '*')
    filenames = sorted(glob(pattern))
    total = 0
    with Pool() as pool:
        it = pool.imap(get_num_samples, filenames)
        for filename, count in zip(filenames, it):
            print(filename, count)
            total += count
    print(total)


if __name__ == '__main__':
    main(parse_args())
