import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.lib.io import file_io
from io import BytesIO

root = '/mnt/property/data/'
raw = root + 'raw'
test = root + 'test'
eval = root + 'eval'
train = root + 'train'


def split_name(file):
    name, ext = os.path.splitext(os.path.basename(file))
    return name, ext


def prune(start=120000, end=132000):
    files = glob.glob(os.path.join(raw, '*.npy'))
    for i, file in enumerate(files):
        arr = np.load(file)[start:end, :]
        name, ext = split_name(file)
        np.save(os.path.join(eval, name + ext), arr)
        print('{:02f}%'.format((i + 1) / 345 * 100))


def to_tfrecords():
    files = glob.glob(os.path.join(eval, '*.npy'))
    for i, file in enumerate(files):
        name, _ = split_name(file)
        with tf.python_io.TFRecordWriter(os.path.join(eval, 'tfrecords', name + '.tfrecords')) as writer:
            images = np.load(file)
            for j in range(images.shape[0]):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(float_list=tf.train.FloatList(value=images[j])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
                }))
                writer.write(example.SerializeToString())
                if (j + 1) % 5000 == 0 and j > 1:
                    print('writing {}th image'.format(j))
        print('wrote {}th class'.format(i))


def make_dataset(params):
    one_hot = tf.one_hot(list(range(345)), 345, dtype=tf.int64)
    features = {'image': tf.FixedLenFeature([784], tf.float32),
                'label': tf.FixedLenFeature([1], tf.int64)}

    def _parse_func(example_proto):
        parsed = tf.parse_single_example(example_proto, features)
        return tf.div(parsed['image'], 255), one_hot[parsed['label'][0]]

    def get_shuffled_dataset(root):
        return tf.data.TFRecordDataset.list_files(os.path.join(root, '*.tfrecords')).repeat() \
            .apply(tf.data.experimental.parallel_interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_func),
                                                            cycle_length=345,
                                                            block_length=10,
                                                            sloppy=True
                                                            )) \
            .shuffle(buffer_size=345*10) \
            .batch(params['batch_size']) \
            .prefetch(1000)

    train_data = get_shuffled_dataset(train)
    eval_data = get_shuffled_dataset(eval)
    # test_data = tf.data.TFRecordDataset.list_files(get_path(test), shuffle=False).interleave()
    return train_data, eval_data


def make_numpy_data(vfold_ratio=0.2):
    # files = file_io.get_matching_files(FLAGS.train_data)
    files = glob.glob(os.path.join('/mnt/vanity/new', '*.npy'))

    x = np.empty([0, 28 ** 2])
    y = np.empty([0])
    class_names = []

    for idx, file in enumerate(files):
        # arr_bytes = file_io.read_file_to_string(file, True)
        # data = np.load(BytesIO(arr_bytes))
        data = np.load(file)
        data = data[0: 1000, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)
        print('Data loading: {:0.2f}%'.format((idx + 1) / 345 * 100))
    data = None
    labels = None

    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    vfold_size = int(x.shape[0] * vfold_ratio)
    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]
    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]

    x_train /= 255.0
    x_test /= 255.0

    y_train = keras.utils.to_categorical(y_train, 345)
    y_test = keras.utils.to_categorical(y_test, 345)
    return x_train, y_train, x_test, y_test, class_names


if __name__ == '__main__':
    prune()
    to_tfrecords()
