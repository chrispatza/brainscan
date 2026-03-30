from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


def build_model(input_shape: tuple[int, int, int]) -> Sequential:
    return Sequential(
        [
            Input(shape=input_shape),
            Conv2D(16, (3, 3), activation="relu", name="conv2d"),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu", name="conv2d_1"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu", name="conv2d_3"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )


def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    val_generator = eval_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    test_generator = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return train_generator, val_generator, test_generator


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_generator, val_generator, test_generator = build_generators()
    model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stopping],
    )

    model.save(MODEL_PATH)

    loss, accuracy = model.evaluate(test_generator)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
