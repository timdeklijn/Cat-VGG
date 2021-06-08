from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
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

    # TODO: add early stopping
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model


def train_model(model, train_generator, valid_generator):
    _ = model.fit(
        train_generator,
        epochs=5,
        validation_data=valid_generator,
    )
    return model


if __name__ == "__main__":
    model = build_model()
    print(model.summary())

    train = Path("data", "processed", "train")
    valid = Path("data", "processed", "valid")
    tg, vg = create_generators(train, valid)

    train_model(model, tg, vg)
