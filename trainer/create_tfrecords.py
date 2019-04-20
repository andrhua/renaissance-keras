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
        help="Path to raw data in numpy arrays format"
    )
    parser.add_argument(
        "--train_dst",
        type=str,
        help="Path to write train data in tfrecords format"
    )
    parser.add_argument(
        "--eval_dst",
        type=str,
        help="Path to write evaluation data in tfrecords format"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of samples per class taken from numpy array for training"
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=2000,
        help="Number of samples per class taken from numpy array for evaluation"
    )
    return parser.parse_known_args()


def to_tfrecords():
    """
    Converts drawings from numpy arrays to .tfrecords format
    """
    def split_name(file):
        """
        Splits filename to a base name and extension.

        :return: tuple consisting of name and extension
        """
        return os.path.splitext(os.path.basename(file))

    def write_example(writer, image, label):
        """
        Serializes example to tfrecord format.

        :param writer: TFRecordWriter
        :param image: list with shape (784), containing pixel values from 0.0 to 1.0
        :param label: index of class to which image belongs
        """
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=image)),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
        }))
        writer.write(example.SerializeToString())

    def convert():
        """
        Reads .npy from disk, takes first FLAGS.train_size elements to form
        a train dataset, then takes next FLAGS.eval_size elements for
        an evaluation dataset.
        """
        files = glob.glob(os.path.join(FLAGS.src, '*.npy'))
        for i, file in enumerate(files):
            name = split_name(file)[0] + '.tfrecords'
            images = np.load(file)
            np.random.shuffle(images)
            with tf.python_io.TFRecordWriter(os.path.join(FLAGS.train_dst, name)) as writer:
                for j in range(FLAGS.train_size):
                    write_example(writer, images[j], [i])
            with tf.python_io.TFRecordWriter(os.path.join(FLAGS.eval.dst, name)) as writer:
                for j in range(FLAGS.train_size, FLAGS.train_size + FLAGS.eval_size):
                    write_example(writer, images[j], [i])

    convert()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    to_tfrecords()
