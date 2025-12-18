import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt

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

### OBS SAVE SOMEWHERE ELSE!!
def save_logmel_examples(ds, classes, out_dir="debug_viz", n=12):
    os.makedirs(out_dir, exist_ok=True)

    # Take n individual examples (not batches) for easy naming
    unbatched = ds.unbatch().take(n)

    for i, (x, y) in enumerate(unbatched):
        x = x.numpy()  # (T, F, 1)
        y = int(y.numpy())

        # Remove channel dim -> (T, F)
        x2 = x[..., 0]

        # Map label to class name if available
        if isinstance(classes, (list, tuple)) and y < len(classes):
            cname = str(classes[y])
        else:
            cname = f"class{y}"

        # Plot
        plt.figure()
        plt.imshow(x2.T, aspect="auto", origin="lower")  # transpose so y-axis = mel bins
        plt.title(f"Example {i} | label={y} | {cname}")
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")

        # Save image
        safe_cname = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in cname)
        png_path = os.path.join(out_dir, f"{i:02d}_label{y}_{safe_cname}.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        # Save raw array too (optional but useful)
        npz_path = os.path.join(out_dir, f"{i:02d}_label{y}_{safe_cname}.npz")
        np.savez_compressed(npz_path, x=x2, y=y, cname=cname)

    print(f"Saved {n} examples to: {out_dir}/")



if __name__ == "__main__":
    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False)
    
    #save_logmel_examples(train_ds, classes, out_dir="debug_viz/train", n=12)
    #save_logmel_examples(val_ds, classes, out_dir="debug_viz/val", n=12)

    model = build_cnn_trad_fpool3(num_classes=len(classes))
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    num_train = len(train_ds)   # or however you build the dataframe
    batch_size = 64  # must match what you used in create_tf_dataset ### OBS GET FROM CONFIG!
    steps_per_epoch = num_train // batch_size
    print(steps_per_epoch) 
    print(num_train)   

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        steps_per_epoch=100,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )