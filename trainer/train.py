import os
import time
from trainer.model import QDModel
from trainer import hypertuning
import tensorflow as tf
from tensorflow.python import keras

if __name__ == "__main__":
    params = hypertuning.get_best_params()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    sess = tf.Session()
    keras.backend.set_session(sess)

    from trainer.data import make_dataset
    train, eval = make_dataset(params)
    m = QDModel(params)
    export_path = os.path.join('export', str(int(time.time())))
    m.train(train, eval, export_path, params)
    m.to_saved_model(export_path)
