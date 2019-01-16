import os
import time
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.saved_model import signature_constants, tag_constants, builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def create_model():
    keras.backend.set_learning_phase(False)

    inputs = layers.Input(shape=(784,), dtype='float32')
    x = layers.Reshape((28, 28, 1))(inputs)
    filters_num = [16, 32, 64]
    for filters in filters_num:
        x = layers.Convolution2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(345, activation='softmax')(x)

    model = keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    return model


def to_saved_model(model, export_path, classes):
    builder = saved_model_builder.SavedModelBuilder(
        os.path.join(export_path, str(int(time.time()))))

    signature = predict_signature_def(
        inputs={'in': model.inputs[0]}, outputs={
            k: v for k, v in zip(classes, model.outputs[0])
        })

    with keras.backend.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            })
        builder.save()
