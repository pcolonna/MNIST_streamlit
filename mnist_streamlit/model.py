""" We will build the model in this module"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

import pprint


def create(input_size):
    """ We will define and return a keras model."""

    model = tf.keras.Sequential()

    model.add(keras.layers.Dense(input_size, activation="relu", input_shape=(input_size,)))

    # The number of hidden unit is arbitrary here. Later or in an other project we will introduce hyperparameter search
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(32, activation="relu"))

    # For our last layer we will use a softmax of size 10, one value per digit
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def train(x_train, y_train, epochs, batch_size):
    """Here we train the model."""

    model = create(input_size=x_train.shape[1])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    return model


def prepare_data(x_train, y_train, x_test, y_test):
    """We will reshape the data so it can be used in a fully connected network"""

    # MNIST contains 60000 images of size 28x28 pixel in the training set
    # So x_train.shape[0] is (60000, 28, 28)

    # Number of training and test examples
    nb_train = x_train.shape[0]
    nb_test = x_test.shape[0]

    # Now we get height and width of the images. Should be 28 for both.
    img_height = x_train.shape[1]
    img_width = x_train.shape[2]

    # And now for the reshaping
    x_train = x_train.reshape((nb_train, img_width * img_height))
    x_test = x_test.reshape((nb_test, img_width * img_height))

    # We also need to convert the labels into their one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def run_experiment(epochs, batch_size):
    """We will load the data, train and report"""

    # We load the dataset from keras directly
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

    model = train(x_train, y_train, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    print(model.metrics_names)
    print(loss, accuracy)


if __name__ == "__main__":

    batch_size = 64
    epochs = 10

    run_experiment(epochs, batch_size)
