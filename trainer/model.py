from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.saved_model import signature_constants, tag_constants, builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


class QDModel:
    def __init__(self, flags):
        keras.backend.set_learning_phase(False)  # workaround to be able to export model later

        inputs = layers.Input(shape=(784,), dtype='float32')
        x = layers.Reshape((28, 28, 1))(inputs)
        filters = [16, 32, 64, 96]
        kernel_size = 3
        pool_size = 2
        for filter in filters:
            x = layers.Convolution2D(filter, (kernel_size, kernel_size), padding='same', activation='relu')(x)
            x = layers.MaxPooling2D((pool_size, pool_size))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=flags.dense_units, activation='relu')(x)
        predictions = layers.Dense(units=flags.classes, activation='softmax')(x)

        model = keras.models.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=Adadelta(),
                      loss=categorical_crossentropy,
                      metrics=['top_k_categorical_accuracy'])
        print(model.summary())

        self.model = model

    def train(self, train, eval, flags):
        tensorboard = TensorBoard(log_dir=flags.logs,
                                  histogram_freq=2,
                                  batch_size=flags.batch_size,
                                  write_graph=True,
                                  write_images=True)
        early_stop = EarlyStopping(patience=2)
        return self.model.fit(train,
                              epochs=flags.epochs,
                              verbose=2,
                              callbacks=[tensorboard, early_stop],
                              validation_data=eval,
                              steps_per_epoch=(flags.classes * flags.train_size) // flags.batch_size,
                              validation_steps=(flags.classes * flags.eval_size) // flags.batch_size,
                              )

    def evaluate(self, test_data):
        score = self.model.evaluate(test_data, verbose=1)
        print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

    def to_saved_model(self, export_path):
        builder = saved_model_builder.SavedModelBuilder(export_path)

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
