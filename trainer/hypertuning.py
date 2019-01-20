import talos as ta
from trainer import data
from trainer.model import QDModel
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.optimizers import Adam, Nadam, Adadelta
from tensorflow.python.keras.losses import categorical_crossentropy


def get_best_params():
    return {
        'dense_units': 512,
        'dense_activation': relu,
        'optimizer': Adam,
        'loss': categorical_crossentropy,
        'epochs': 5,
        'batch_size': 256,
        'train_samples_per_class': 900,
        'eval_samples_per_class': 100
    }


p = {
    'dense_units': [128, 256, 512, 1024],
    'dense_activation': [relu],
    'optimizer': [Adam, Nadam, Adadelta],
    'loss': [categorical_crossentropy],
    'epochs': [5, 10, 15, 20, 25],
    'batch_size': [8, 64, 256],
    'dropout': [None],
    'weight_regulizer': [None],
    'train_samples_per_class': 900,
    'eval_samples_per_class': 100
}


def my_model(params):
    m = QDModel(params)
    out = m.train(x_train, y_train, params)
    return out, m.model


if __name__ == "__main__":
    x_train, y_train = data.make_numpy_data()
    h = ta.Scan(x_train, y_train,
                params=p,
                model=my_model,
                dataset_name='qd',
                experiment_no='1',
                grid_downsample=.5)
