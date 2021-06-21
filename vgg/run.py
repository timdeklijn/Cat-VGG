from pathlib import Path

import numpy as np
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import accuracy_score, balanced_accuracy_score

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
    model.add(Dropout(0.5, name="dropout1"))
    model.add(Dense(4096, name="fc2"))
    model.add(Dropout(0.5, name="dropout2"))

    # Output layer
    model.add(Dense(2, activation="softmax", name="predictions"))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(from_logits=True)])

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

    # Calculate and log metrics to W&B
    wandb.log({"accuracy": accuracy_score(y_true, y_pred)})
    wandb.log({"balanced accuracy": balanced_accuracy_score(y_true, y_pred)})

    # Log a confusion matrix to W&B
    wandb.log({
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=["Maz", "Rey"])})


def train_model(model, train_generator, valid_generator):
    """
    Prepare tracking on MLFlow server, train a model and log the metrics
    """
    # Create a small set to visualize predictions on and show in W&B
    val_images, val_labels = [], []
    for _ in range(2):
        v = valid_generator.next()
        val_images.extend(v[0])
        val_labels.extend(v[1])

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001)

    # Train the model, we do not need to get the history since this is
    # auto logged by MLFlow.
    _ = model.fit(
        train_generator,
        epochs=2000,
        validation_data=valid_generator,
        callbacks=[
            reduce_lr,
            WandbCallback(
                data_type="image",
                training_data=(val_images, val_labels),
                labels=["Maz", "Rey"])])

    # Calculate model metrics based on predictions on the validation set.
    calculate_model_metrics(model, valid_generator)


if __name__ == "__main__":
    wandb.init(project="cat-vgg")
    model = build_model()

    train = Path("data", "processed", "train")
    valid = Path("data", "processed", "valid")
    tg, vg = create_generators(train, valid)

    train_model(model, tg, vg)
