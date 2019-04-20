import argparse
import os
import time
from trainer.model import QDModel
import tensorflow as tf
from tensorflow.python import keras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes",
        type=int,
        default=345,
        help="Number of classes in classification problem"
    )
    parser.add_argument(
        "--dense_units",
        type=int,
        default=512,
        help="Number of units in dense classification layer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of samples per class for training"
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=2000,
        help="Number of samples per class for evaluation"
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs",
        help="Directory to write Tensorboard logs"
    )
    return parser.parse_known_args()


def make_datasets():
    """
    Creates train and evaluation tf.data.Datasets from tfrecords files specified in input arguments.

    :return: pair of train and evaluation datasets
    """
    one_hot = tf.one_hot(list(range(flags.classes)), flags.classes, dtype=tf.int64)
    features = {'image': tf.FixedLenFeature([784], tf.float32),
                'label': tf.FixedLenFeature([1], tf.int64)}

    def _parse_func(example_proto):
        parsed = tf.parse_single_example(example_proto, features)
        return tf.div(parsed['image'], 255), one_hot[parsed['label'][0]]

    def get_dataset(root, shuffle=True):
        data = tf.data.TFRecordDataset.list_files(os.path.join(root, '*.tfrecords')).repeat()\
            .apply(tf.data.experimental.parallel_interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_func),
                                                            cycle_length=flags.classes,
                                                            block_length=10,
                                                            sloppy=True
                                                            ))
        if shuffle:
            data = data.shuffle(buffer_size=flags.classes * 10)
        return data\
            .batch(flags.batch_size)\
            .prefetch(1000)

    return get_dataset(flags.train_src), get_dataset(flags.eval_src, False)


if __name__ == "__main__":
    flags, unparsed = parse_args()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    sess = tf.Session()
    keras.backend.set_session(sess)

    train, eval = make_datasets()
    model = QDModel(flags)
    export_path = os.path.join('export', str(int(time.time())))
    model.train(train, eval, flags)
    model.to_saved_model(export_path)
