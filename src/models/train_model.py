import os
import cv2

# import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path


from model_params import (
    CLASSES_LIST,
    DATASET_DIR,
    PREPROCESSED_DIR,
    SEQUENCE_LENGTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

# from moviepy.editor import *
# %matplotlib inline

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


def create_convlstm_model():
    """
    This function will construct the required convlstm model.
    Returns:
        model: It is the required constructed convlstm model.
    """

    # We will use a Sequential model for model construction
    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################

    model.add(
        ConvLSTM2D(
            filters=4,
            kernel_size=(3, 3),
            activation="tanh",
            data_format="channels_last",
            recurrent_dropout=0.2,
            return_sequences=True,
            input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        )
    )

    model.add(
        MaxPooling3D(pool_size=(1, 2, 2), padding="same", data_format="channels_last")
    )
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(
        ConvLSTM2D(
            filters=8,
            kernel_size=(3, 3),
            activation="tanh",
            data_format="channels_last",
            recurrent_dropout=0.2,
            return_sequences=True,
        )
    )

    model.add(
        MaxPooling3D(pool_size=(1, 2, 2), padding="same", data_format="channels_last")
    )
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(
        ConvLSTM2D(
            filters=14,
            kernel_size=(3, 3),
            activation="tanh",
            data_format="channels_last",
            recurrent_dropout=0.2,
            return_sequences=True,
        )
    )

    model.add(
        MaxPooling3D(pool_size=(1, 2, 2), padding="same", data_format="channels_last")
    )
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(
        ConvLSTM2D(
            filters=16,
            kernel_size=(3, 3),
            activation="tanh",
            data_format="channels_last",
            recurrent_dropout=0.2,
            return_sequences=True,
        )
    )

    model.add(
        MaxPooling3D(pool_size=(1, 2, 2), padding="same", data_format="channels_last")
    )
    # model.add(TimeDistributed(Dropout(0.2)))

    model.add(Flatten())

    model.add(Dense(len(CLASSES_LIST), activation="softmax"))

    ########################################################################################################################

    # Display the models summary.
    model.summary()

    # Return the constructed convlstm model.
    return model


def create_LRCN_model():
    """
    This function will construct the required LRCN model.
    Returns:
        model: It is the required constructed LRCN model.
    """

    # We will use a Sequential model for model construction.
    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################

    model.add(
        TimeDistributed(
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        )
    )

    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    # model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(32))

    model.add(Dense(len(CLASSES_LIST), activation="softmax"))

    ########################################################################################################################

    # Display the models summary.
    model.summary()

    # Return the constructed LRCN model.
    return model


def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    """
    This function will plot the metrics passed to it in a graph.
    Args:
        model_training_history: A history object containing a record of training and validation
                                loss values and metrics values at successive epochs
        metric_name_1:          The name of the first metric that needs to be plotted in the graph.
        metric_name_2:          The name of the second metric that needs to be plotted in the graph.
        plot_name:              The title of the graph.
    """

    # Get metric values using metric names as identifiers.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_value_1))

    # Plot the Graph.
    plt.plot(epochs, metric_value_1, "blue", label=metric_name_1)
    plt.plot(epochs, metric_value_2, "red", label=metric_name_2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()


def load_data():
    directory = Path("data/preprocessed")
    features_train = np.load(directory / "features_train.npy")
    labels_train = np.load(directory / "labels_train.npy")
    features_test = np.load(directory / "features_test.npy")
    labels_test = np.load(directory / "labels_test.npy")
    return features_train, labels_train, features_test, labels_test


if __name__ == "__main__":
    features_train, labels_train, features_test, labels_test = load_data()

    use_LSTM = True

    if use_LSTM:
        model = create_convlstm_model()
        # Create an Instance of Early Stopping Callback
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", restore_best_weights=True
        )

        # Compile the model and specify loss function, optimizer and metrics values to the model
        model.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )

        # Start training the model.
        convlstm_model_training_history = model.fit(
            x=features_train,
            y=labels_train,
            epochs=50,
            batch_size=4,
            shuffle=True,
            validation_split=0.2,
            callbacks=[early_stopping_callback],
        )

        # Evaluate model:
        # Evaluate the trained model.
        model_evaluation_history = model.evaluate(features_test, labels_test)

        # Save

        # Plot result
        # Visualize the training and validation accuracy metrices.
        plot_metric(
            convlstm_model_training_history,
            "accuracy",
            "val_accuracy",
            "Total Accuracy vs Total Validation Accuracy",
        )
