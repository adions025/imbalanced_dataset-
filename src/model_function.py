from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
import tensorflow as tf

LR = 0.001
BATCH_SIZE = 512


def compile_model(model, optimizer, loss, lr=LR):
    """
    Esta función recibe por parámetro el modelo a compilar y además el optimizador
    y la función de perdida, por defecto están definas las que se necesitan para
    esta practica. También podría recibir el learning rate aunque por defecto es 1-e2

    :param model:
    :param optimizer:
    :param loss:
    :param lr:
    :return: a compiled model
    """
    if optimizer == 'Adam':
        opt = optimizers.Adam(lr=lr)
    if optimizer == 'rms':
        opt = optimizers.RMSprop(lr=lr)
    if optimizer == 'SGD':
        opt = optimizers.SGD(lr=lr)
    if optimizer == 'Adadelta':
        opt = optimizers.Adadelta(lr=lr)
    if optimizer == 'Adagrad':
        opt = optimizers.Adagrad(lr=lr)

    if loss == 'sparse_c_c':
        los = 'sparse_categorical_crossentropy'
    if loss == 'categorial_c':
        los = tf.keras.losses.CategoricalCrossentropy()

    print(f'model compile using {opt, los, lr}')
    model.compile(optimizer=opt, loss=los, metrics=['accuracy'])
    return model


METRIC = "val_loss"
PATH = './model.h5'


def create_callbacks(metric=METRIC, path=PATH):
    """
    Function para crear la callbacks del modelo, permite pasar un filepath nuevo
    para cada modelo, por defecto la métrica de monitorización es la val_loss.
    :param metric:
    :param path:
    :return:
    """
    checkpoint = ModelCheckpoint(
        filepath=path,
        monitor=metric,
        mode='max',
        save_best_only=True,
        verbose=1)
    callbacks = [checkpoint]
    return callbacks


def create_model(neurons, dim, classes):
    """
    Function to create model
    :param neurons:
    :param dim:
    :param classes:
    :return: a model
    """
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=dim, activation='relu'))
    [model.add(Dense(neurons[i], activation='relu')) for i in range(1, len(neurons))]
    model.add(Dense(classes, activation='softmax'))
    return model
