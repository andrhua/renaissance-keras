import os
import time
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.saved_model import signature_constants, tag_constants, builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


class QDModel:
    def __init__(self, params):
        keras.backend.set_learning_phase(False)

        inputs = layers.Input(shape=(784,), dtype='float32')
        x = layers.Reshape((28, 28, 1))(inputs)
        filters_num = [16, 32, 64]
        for filters in filters_num:
            x = layers.Convolution2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=params['dense_units'], activation=params['dense_activation'])(x)
        predictions = layers.Dense(units=345, activation='softmax')(x)

        model = keras.models.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=params['optimizer'](),
                      loss=params['loss'],
                      metrics=['top_k_categorical_accuracy'])
        print(model.summary())
        self.model = model

    def train(self, train, eval, params):
        tensorboard = TensorBoard(histogram_freq=2,
                                  batch_size=params['batch_size'],
                                  write_graph=True,
                                  write_images=True)
        early_stop = EarlyStopping(patience=2)
        return self.model.fit(train,
                              epochs=params['epochs'],
                              verbose=2,
                              callbacks=[tensorboard, early_stop],
                              validation_data=eval,
                              steps_per_epoch=(345 * params['train_samples_per_class']) // params['batch_size'],
                              validation_steps=(345 * params['eval_samples_per_class']) // params['batch_size'],
                              )

    def evaluate(self, test_data):
        score = self.model.evaluate(test_data, verbose=1)
        print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

    def to_saved_model(self, export_path):
        builder = saved_model_builder.SavedModelBuilder(
            os.path.join(export_path, str(int(time.time()))))

        signature = predict_signature_def(
            inputs={'in': self.model.inputs[0]},
            outputs={'out': self.model.outputs[0]})

        with keras.backend.get_session() as sess:
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
                })
            builder.save()
