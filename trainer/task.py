import os
import glob
import numpy as np
import tensorflow as tf
import trainer.model as M
import trainer.util as U
from io import BytesIO
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python import keras
from tensorflow.python.lib.io import file_io


FLAGS = None
NUM_CLASSES = 345


def load(root='/mnt/vanity/data', samples=40000):
    new = os.path.join(root, 'new')
    files = glob.glob(os.path.join(root, '*.npy'))
    for file in files:
        data = np.load(file)
        data = data[0: samples, :]
        np.save(os.path.join(new, os.path.basename(file)), data)


def load_and_process(vfold_ratio=0.2):
    files = file_io.get_matching_files(FLAGS.train_data)
    # files = glob.glob(os.path.join('/mnt/vanity/data', '*.npy'))

    # initialize variables
    x = np.empty([0, 28**2])
    y = np.empty([0])
    class_names = []

    # load a subset of the data to memory
    for idx, file in enumerate(files):
        arr_bytes = file_io.read_file_to_string(file, True)
        data = np.load(BytesIO(arr_bytes))
        data = data[0: FLAGS.data_samples, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)
        print('Data loading: {:0.2f}%'.format((idx + 1) / NUM_CLASSES * 100))
    data = None
    labels = None

    # separate into training and testing
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

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test, class_names


def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=3,
        verbose=1
    )

    model.fit(x=x_train, y=y_train,
              validation_split=0.1,
              batch_size=FLAGS.batch_size,
              verbose=2,
              epochs=FLAGS.num_epochs,
              callbacks=[stopping])

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy: {:0.2f}%'.format(score[1] * 100))


if __name__ == "__main__":
    FLAGS = U.parse_args()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    sess = tf.Session()
    keras.backend.set_session(sess)

    x_train, y_train, x_test, y_test, classes = load_and_process()
    model = M.create_model()
    train_and_evaluate(model, x_train, y_train, x_test, y_test)
    M.to_saved_model(model, os.path.join(FLAGS.job_dir, 'export'), classes)
