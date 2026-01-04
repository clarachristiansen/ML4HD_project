# test different vision areas e.g. horizontal vs vertical
# autoencoder to learn compressed representations + SVM on top
# transformer-based models with CNN
import tensorflow as tf
from tensorflow.keras import layers, models
from .preprocessing import get_datasets
from .utils_plotting import save_logmel_examples


def build_cnn_time(input_shape=(32, 40, 1), num_classes=12):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=(9, 3), padding="same", activation="relu")(inputs)  # long in time
    x = layers.MaxPool2D(pool_size=(2, 1))(x)  # pool time only

    x = layers.Conv2D(64, kernel_size=(9, 3), padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 1))(x)  # pool time only

    x = layers.GlobalMaxPooling2D()(x)  # one vector per utterance
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_time")

    return model



def build_cnn_freq(input_shape=(32, 40, 1), num_classes=12):
    inputs = layers.Input(shape=input_shape)

    
    x = layers.Conv2D(32, kernel_size=(3, 9), padding="same", activation="relu")(inputs)  # long in freq
    x = layers.MaxPool2D(pool_size=(2, 1))(x)  # SAME pooling as time model (important!)

    x = layers.Conv2D(64, kernel_size=(3, 9), padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 1))(x)  # SAME pooling

    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_freq")

    return model

if __name__ == "__main__":
    frames = 98
    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False, frames=frames)

    model_time = build_cnn_time(num_classes=len(classes), input_shape=(frames, 40, 1))
    model_time.summary()

    model_time.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    ) 

    history = model_time.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )
    print("Time-based CNN model trained.")
    print("Evaluating on test set:")
    model_time.evaluate(test_ds)

    model_freq = build_cnn_freq(num_classes=len(classes), input_shape=(frames, 40, 1))
    model_freq.summary()

    model_freq.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history = model_freq.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )
    print("Freq-based CNN model trained.")
    print("Evaluating on test set:")
    model_freq.evaluate(test_ds)
