from imports import *


def plot_cm(cm: confusion_matrix, labels: list):
    """
    Function to plot a confusion matrix. Se necesita a
    computed confusion matrix object. By definition a
    confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in
    group :math:`i` and  predicted to be in group :math:`j`.

    :param cm: A computed matrix object
    :param labels: A list of unique target values
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white",
                 fontsize=12)

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_accuracy(history):
    """
    Function to plot el accuracy del set de train y validation
    Is needed a `History` object. Its `History.history`
    attribute is a record of training loss values and metrics
    values at successive epochs, as well as validation loss
    values and validation metrics values (if applicable).

    :param history: a `History` object
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train val', 'Test val'], loc='upper right')
    plt.show()


def plot_loss(history):
    """
    Function to plot el loss del set de train y validation
    Is needed a `History` object. Its `History.history`
    attribute is a record of training loss values and metrics
    values at successive epochs, as well as validation loss
    values and validation metrics values (if applicable).

    :param history: a `History` object
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train loss', 'Test loss'], loc='upper right')
    plt.show()
