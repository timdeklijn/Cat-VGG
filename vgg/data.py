from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators(train_path: Path, validation_path: Path):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.2, 0.8),
        horizontal_flip=False)

    # The validation set will NEVER be augmented.
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Generators now point to the complete dataset
    train_generator = train_datagen.flow_from_directory(
        train_path, target_size=(224, 224), batch_size=10)

    validation_generator = test_datagen.flow_from_directory(
        validation_path, target_size=(224, 224), batch_size=10)

    return train_generator, validation_generator
