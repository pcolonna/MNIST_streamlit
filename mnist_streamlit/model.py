""" We will build the model in this module"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint 
from tensorflow.keras.models import load_model

import pprint
import io

import plot
import numpy as np

import random

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



def prepare_data():
    """We will reshape the data so it can be used in a fully connected network"""


    # MNIST contains 60000 images of size 28x28 pixel in the training set
    # So x_train.shape[0] is (60000, 28, 28)

    # We load the dataset from keras directly
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

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


def run_experiment(epochs, batch_size, progress_bar, status_text, val_acc_text, chart):
    """We will load the data, train and report"""

    x_train, y_train, x_test, y_test = prepare_data()

    model = create(input_size=x_train.shape[1])

    mc = ModelCheckpoint("best_model.h5", monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[mc, myCallback(epochs, progress_bar, status_text, val_acc_text, chart)])
    
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1, )
    
    summarize(model)

    print(model.metrics_names)
    print(loss, accuracy)


def summarize(model):
    """ Returns the summary fo the model to streamlit"""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def predict(missing_model_text, canvas, ground_truth_text, prediction_text):
    """ Take the trained model, pick an example at random and returns its prediction."""

    #try:
    if True:
        missing_model_text.text('')
        best_model = load_model("best_model.h5")

        _, _, x_test, y_test = prepare_data()

        random_idx = random.randint(0, len(x_test))

        random_mnist_example = y_test[random_idx]
        random_mnist_input = x_test[random_idx].reshape(1, 784)

        # We now predict the number
        y_pred = best_model.predict(random_mnist_input)
        canvas.image(random_mnist_input.reshape(28,28), width=150)


        ground_truth_text.markdown(f"** Actual Number:** {np.argmax(random_mnist_example)}")
        prediction_text.markdown(f"** Predicted Number:** {np.argmax(y_pred)}")

    #except IOError as e:
    #    missing_model_text.markdown("** You need to click on Train Model first **")
    #except Exception as e:
    #    print(e)

def predict_from_drawing(missing_model_text, img,  prediction_text):
    """ Take the trained model, pick an example at random and returns its prediction."""

    #try:
    if True:
        missing_model_text.text('')
        best_model = load_model("best_model.h5")

        

       
        mnist_input = img.reshape(1, 784)

        # We now predict the number
        y_pred = best_model.predict(mnist_input)

        

        prediction_text.markdown(f"** Predicted Number:** {np.argmax(y_pred)}")

    #except IOError as e:
    #    missing_model_text.markdown("** You need to click on Train Model first **")
    #except Exception as e:
    #    print(e)

class myCallback(keras.callbacks.Callback):

    def __init__(self, max_epochs, progress_bar, status_text, val_acc_text, chart):
        self.max_epochs = max_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.chart = chart
        self.loss = []
        self.val_acc_text = val_acc_text
        self.best_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1)  / self.max_epochs)
        self.status_text.text(f"{round((epoch + 1)/ self.max_epochs * 100) }% Complete")
        
        accuracy = logs['accuracy']
        self.chart.add_rows(np.array([[accuracy]]))

        best_val_accuracy = logs['val_accuracy'] if logs['val_accuracy'] > self.best_val_accuracy else self.val_accuracy
        self.val_acc_text.text(f"Best Validation Accuracy: {round(best_val_accuracy, 4)}")
