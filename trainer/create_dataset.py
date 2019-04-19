import argparse
import glob
import os
import tensorflow as tf
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        help="Path to raw data in numpy arrays format")
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to write data in tfrecords format")
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples per class taken from numpy array")
    return parser.parse_known_args()


def to_tfrecords():
    def split_name(file):
        return os.path.splitext(os.path.basename(file))

    files = glob.glob(os.path.join(FLAGS.src, '*.npy'))
    for i, file in enumerate(files):
        name = split_name(file)[0]
        with tf.python_io.TFRecordWriter(os.path.join(FLAGS.dst, name + '.tfrecords')) as writer:
            images = np.load(file)
            np.random.shuffle(images)
            for j in range(FLAGS.samples):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(float_list=tf.train.FloatList(value=images[j])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
                }))
                writer.write(example.SerializeToString())
        print('Converted {}th class'.format(i))


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    to_tfrecords()
