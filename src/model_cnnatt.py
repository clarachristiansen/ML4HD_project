# test different vision areas e.g. horizontal vs vertical
# autoencoder to learn compressed representations + SVM on top
# transformer-based models with CNN
import tensorflow as tf
from tensorflow.keras import layers, models
from .preprocessing import get_datasets
from .utils_plotting import save_logmel_examples


def build_cnn_horizontal(input_shape=(32, 40, 1), num_classes=12):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(
        filters=64,
        kernel_size=(40, 16),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv1",
    )(inputs)

    # Max-pool only in frequency: pool_size=(3,1)
    x = layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="pool1_freq",
    )(x)

    # Conv2: m=10, r=4, n=128, no pooling
    x = layers.Conv2D(
        filters=128,
        kernel_size=(16, 8),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv2",
    )(x)

    # Flatten
    x = layers.Flatten(name="flatten")(x)

    # Non-linear DNN layer: 512 ReLU
    x = layers.Dense(512, activation="relu", name="dnn_512")(x)

    x = layers.Dense(256, activation="relu", name="dnn_256")(x)

    x = layers.Dense(64, activation="relu", name="dnn_64")(x)

    # Softmax output
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_horizontal")

    return model

if __name__ == "__main__":
    frames = 98
    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False, frames=frames)
    
    save_logmel_examples(train_ds, classes, out_dir="debug_viz/train", n=12)
    save_logmel_examples(val_ds, classes, out_dir="debug_viz/val", n=12)

    #model = build_cnn_trad_fpool3(num_classes=len(classes))
    model = build_cnn_horizontal(num_classes=len(classes), input_shape=(frames, 40, 1))
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    ) 

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )