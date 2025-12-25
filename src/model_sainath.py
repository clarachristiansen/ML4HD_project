import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
from .utils_plotting import save_logmel_examples
from .preprocessing import get_datasets

def build_cnn_trad_fpool3(input_shape=(32, 40, 1), num_classes=12):
    """
    Sainath-style cnn-trad-fpool3 for keyword spotting.

    input_shape: (time_frames, mel_bins, 1) = (32, 40, 1)
    num_classes: number of keyword targets (incl. unknown/silence as needed)
    """

    inputs = layers.Input(shape=input_shape)

    # Conv1: m=20 (time), r=8 (freq), n=64, pool q=3 in frequency
    x = layers.Conv2D(
        filters=64,
        kernel_size=(20, 8),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv1",
    )(inputs)

    # Max-pool only in frequency: pool_size=(1,3)
    x = layers.MaxPool2D(
        pool_size=(1, 3),
        strides=(1, 3),
        padding="valid",
        name="pool1_freq",
    )(x)

    # Conv2: m=10, r=4, n=64, no pooling
    x = layers.Conv2D(
        filters=64,
        kernel_size=(10, 4),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv2",
    )(x)

    # Flatten
    x = layers.Flatten(name="flatten")(x)

    # Linear low-rank layer: 32 units (no nonlinearity in the paper; we mimic with linear)
    x = layers.Dense(32, activation=None, name="linear_low_rank")(x)

    # Non-linear DNN layer: 128 ReLU
    x = layers.Dense(128, activation="relu", name="dnn_128")(x)

    # Softmax output
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_trad_fpool3")

    return model

def build_cnn_tpool2(input_shape=(32, 40, 1), num_classes=12):
    """
    Sainath & Parada (2015) 'cnn-tpool2' style CNN.
    Adapted to full-utterance log-mel input: (98 frames, 40 mel bins, 1 channel).

    Paper (Table 5):
      Conv1: m=21, r=8, n=94 ; pool in time p=2 and freq q=3
      Conv2: m=6,  r=4, n=94 ; no pooling
      Linear: 32
    Then (as in your baseline): Dense 128 ReLU -> Softmax
    """
    inputs = layers.Input(shape=input_shape)

    # Conv1: (21 x 8), 94 feature maps
    x = layers.Conv2D(
        filters=94,
        kernel_size=(21, 8),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv1",
    )(inputs)

    # Pool in time by p=2, in freq by q=3 (non-overlapping)
    x = layers.MaxPool2D(
        pool_size=(2, 3),
        strides=(2, 3),
        padding="valid",
        name="pool_t2_f3",
    )(x)

    # Conv2: (6 x 4), 94 feature maps
    x = layers.Conv2D(
        filters=94,
        kernel_size=(6, 4),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv2",
    )(x)

    # Flatten + low-rank linear layer (32)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(32, activation=None, name="linear_32")(x)

    # Nonlinear DNN layer (consistent with your previous implementation)
    x = layers.Dense(128, activation="relu", name="dnn_128")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_tpool2")

    return model

if __name__ == "__main__":
    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False)
    
    save_logmel_examples(train_ds, classes, out_dir="debug_viz/train", n=12)
    save_logmel_examples(val_ds, classes, out_dir="debug_viz/val", n=12)

    #model = build_cnn_trad_fpool3(num_classes=len(classes))
    model = build_cnn_tpool2(num_classes=len(classes))
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