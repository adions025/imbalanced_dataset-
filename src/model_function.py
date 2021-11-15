from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import optimizers
import tensorflow as tf


def create_model(neurons: list, dim: int, classes: int) -> Model:
    """
    Function to create a model, se necesita una lista de capas ocultas.
    Se necesita el tamaño de la entrada y la salida, por defecto la
    las capas ocultas se activan con relu, y la capa de salida softmax.
    Cause we have a multi-class classification problem = there is only
    one "right answer" = the outputs are mutually exclusive, then we
    use a softmax function. The softmax will enforce that the sum of
    the probabilities of  output classes are equal to one, so in order
    to increase the probability of a particular class.

    :param neurons: A list of hidden layers
    :param dim: A int value of input dimension
    :param classes: A int value of output dimension
    :return: a model A tf model
    """
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=dim, activation='relu'))
    [model.add(Dense(neurons[i], activation='relu')) for i in range(1, len(neurons))]
    model.add(Dense(classes, activation='softmax'))
    return model


LR = 0.001
BATCH_SIZE = 512


def compile_model(model: Model, optimizer: str, loss: str, lr: float = LR) -> Model:
    """
    Esta función recibe por parámetro el modelo a compilar y además el optimizador
    y la función de perdida, por defecto se definen ciertos parametros. Por ejemplo
    el learning rate aunque por defecto es 1-e2

    :param model: A tf model already created
    :param optimizer: A str optimizer
    :param loss: A str loss
    :param lr: A float
    :return: A compiled model
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


def create_callbacks(metric: str = METRIC, path: str = PATH) -> list:
    """
    Function para crear la callbacks del modelo, permite pasar un filepath nuevo
    para cada modelo, por defecto la métrica de monitorización es la val_loss.

    :param metric: A str function loss name
    :param path: A str like /to/path/
    :return: a callback ModelCheckpoint
    """
    checkpoint = ModelCheckpoint(
        filepath=path,
        monitor=metric,
        mode='max',
        save_best_only=True,
        verbose=1)
    callbacks = [checkpoint]
    return callbacks
