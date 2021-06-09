from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import mlflow
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                             accuracy_score,
                             balanced_accuracy_score)

from data import create_generators


def build_model():
    model = tf.keras.Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same",
                     name="block1_conv1", input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation="relu",
                     padding="same", name="block1_conv2"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool"))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation="relu",
                     padding="same", name="block2_conv1"))
    model.add(Conv2D(128, (3, 3), activation="relu",
                     padding="same", name="block2_conv2"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool"))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation="relu",
                     padding="same", name="block3_conv1"))
    model.add(Conv2D(256, (3, 3), activation="relu",
                     padding="same", name="block3_conv2"))
    model.add(Conv2D(256, (3, 3), activation="relu",
                     padding="same", name="block3_conv3"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool"))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation="relu",
                     padding="same", name="block4_conv1"))
    model.add(Conv2D(512, (3, 3), activation="relu",
                     padding="same", name="block4_conv2"))
    model.add(Conv2D(512, (3, 3), activation="relu",
                     padding="same", name="block4_conv3"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool"))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation="relu",
                     padding="same", name="block5_conv1"))
    model.add(Conv2D(512, (3, 3), activation="relu",
                     padding="same", name="block5_conv2"))
    model.add(Conv2D(512, (3, 3), activation="relu",
                     padding="same", name="block5_conv3"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool"))

    # Connected layers
    model.add(Flatten(name="flatten"))
    model.add(Dense(4096, name="fc1"))
    model.add(Dense(4096, name="fc2"))

    # Output layer
    model.add(Dense(2, activation="softmax", name="predictions"))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model


def calculate_model_metrics(model, valid_generator):
    """
    Get the predictions for the validation set and calculate model metrics
    """
    # Extract labels from the generator
    y_true = valid_generator.classes

    # Get predictions from model
    predictions = model.predict(valid_generator)
    y_pred = np.argmax(predictions, axis=1)

    # Calculate and log metrics
    mlflow.log_metric(key="accuracy", value=accuracy_score(y_true, y_pred))
    mlflow.log_metric(key="balanced accuracy",
                      value=balanced_accuracy_score(y_true, y_pred))

    # Create, save and log a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=[
                           "Maz", "Rey"]).plot(values_format="d")
    cm_name = Path("figures", "cm.png")
    plt.savefig(cm_name)
    mlflow.log_artifact(cm_name)


def train_model(model, train_generator, valid_generator):
    """
    Prepare tracking on MLFlow server, train a model and log the metrics
    """

    # Setup remote server
    remote_server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("cats-vgg16")

    # Setup auto logging
    mlflow.tensorflow.autolog()

    # Start training
    with mlflow.start_run():

        # Early stopping based on validation loss.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3)

        # Train the model, we do not need to get the history since this is auto logged
        # by MLFlow.
        _ = model.fit(
            train_generator,
            epochs=2,
            validation_data=valid_generator,
            callbacks=[early_stopping])

        # Calculate model metrics based on predictions on the validation set.
        calculate_model_metrics(model, valid_generator)


if __name__ == "__main__":
    model = build_model()
    print(model.summary())

    train = Path("data", "processed", "train")
    valid = Path("data", "processed", "valid")
    tg, vg = create_generators(train, valid)

    train_model(model, tg, vg)
