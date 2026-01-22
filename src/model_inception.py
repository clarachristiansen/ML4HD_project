import tensorflow as tf
from tensorflow.keras import layers, models
from .preprocessing import get_datasets

def inception_block(inputs, filters, name='inception_block'):
    inception_1x1 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu",
                       name=f"{name}_1x1")(inputs)

    inception_3x3_1x1 = layers.Conv2D(filters * 1.5, (1, 1), padding="same", activation="relu",
                       name=f"{name}_3x3_1x1")(inputs)
    inception_3x3 = layers.Conv2D(filters * 2, (3, 3), padding="same", activation="relu",
                       name=f"{name}_3x3")(inception_3x3_1x1)

    inception_5x5_1x1 = layers.Conv2D(filters * 0.5, (1, 1), padding="same", activation="relu",
                       name=f"{name}_5x5_1x1")(inputs)
    inception_5x5 = layers.Conv2D(filters, (5, 5), padding="same", activation="relu",
                          name=f"{name}_5x5")(inception_5x5_1x1)

    inception_pool = layers.MaxPool2D((3, 3), strides=(1, 1), padding="same",
                          name=f"{name}_pool")(inputs)
    inception_pool_1x1 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu",
                       name=f"{name}_pool_1x1")(inception_pool)

    x = layers.concatenate([inception_1x1, inception_3x3, inception_5x5, inception_pool_1x1],
                           axis=-1, name=f"{name}_concat")
    return x

def build_cnn_inception_1(input_shape=(32, 40, 1), num_classes=12):
    """
    Inception-style CNN for keyword spotting.

    input_shape: (time_frames, mel_bins, 1) = (32, 40, 1)
    num_classes: number of keyword targets (incl. unknown/silence as needed)

    """

    inputs = layers.Input(shape=input_shape)

    # Conv1: m=7 (time), r=7 (freq), n=64, pool q=3 in frequency
    x = layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv1",
    )(inputs)

    # Max-pool pool_size=(2,2)
    x = layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="pool1",
    )(x)

    # Conv2: m=1(time), r=1 (freq), n=64, pool q=3 in frequency
    x = layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv2_1x1",
    )(x)

    # Conv3: m=3(time), r=3 (freq), n=128, pool q=3 in frequency
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv2_3x3",
    )(x)

    x = layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="pool2",
    )(x)


    x = inception_block(x, filters=32, name='inception_1')

    x = layers.AveragePooling2D((3, 3), strides=3)(x)
    x = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dnn_128")(x)
    x = layers.Dropout(0.5)(x)
    # Softmax output
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_1")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_inception")

    return model


### MEDIUM

def build_cnn_inception_2(input_shape=(32, 40, 1), num_classes=12):
    """
    Inception-style CNN for keyword spotting.

    input_shape: (time_frames, mel_bins, 1) = (32, 40, 1)
    num_classes: number of keyword targets (incl. unknown/silence as needed)

    """

    inputs = layers.Input(shape=input_shape)

    # Conv1: m=7 (time), r=7 (freq), n=64, pool q=3 in frequency
    x = layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv1",
    )(inputs)

    # Max-pool pool_size=(2,2)
    x = layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="pool1",
    )(x)

    # Conv2: m=1(time), r=1 (freq), n=64, pool q=3 in frequency
    x = layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv2_1x1",
    )(x)

    # Conv3: m=3(time), r=3 (freq), n=128, pool q=3 in frequency
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv2_3x3",
    )(x)

    x = layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="pool2",
    )(x)


    x = inception_block(x, filters=32, name='inception_1')

    x1 = layers.AveragePooling2D((3, 3), strides=3)(x)
    x1 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = layers.Flatten(name="flatten1")(x1)
    x1 = layers.Dense(128, activation="relu", name="dnn_128")(x1)
    x1 = layers.Dropout(0.5)(x1)
    # Softmax output
    outputs1 = layers.Dense(num_classes, activation="softmax", name="softmax_1")(x1)

    x = inception_block(x, filters=64, name='inception_2')
    
    x2 = layers.AveragePooling2D((3, 3), strides=3)(x)
    x2 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x2)
    x2 = layers.Flatten(name="flatten2")(x2)
    x2 = layers.Dense(256, activation="relu", name="dnn_256")(x2)
    x2 = layers.Dropout(0.5)(x2)
    # Softmax output
    outputs2 = layers.Dense(num_classes, activation="softmax", name="softmax_2")(x2)

    model = models.Model(inputs=inputs, outputs=[outputs1, outputs2], name="cnn_inception")

    return model


if __name__ == "__main__":
    frames = 98
    size = 'small'
    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False, frames=frames)

    if size == 'small':
        model_inception = build_cnn_inception_1(num_classes=len(classes), input_shape=(frames, 40, 1))
        model_inception.summary()

        model_inception.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={'softmax_1': tf.keras.losses.SparseCategoricalCrossentropy()},
            loss_weights={"softmax_1": 1.0},
            metrics={"softmax_1": ["accuracy"]},
        )

        history = model_inception.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
        )
    elif size == 'medium':
        model_inception = build_cnn_inception_2(num_classes=len(classes), input_shape=(frames, 40, 1))
        model_inception.summary()

        model_inception.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={'softmax_1': tf.keras.losses.SparseCategoricalCrossentropy(),
                  'softmax_2': tf.keras.losses.SparseCategoricalCrossentropy()},
            loss_weights={
                "softmax_1": 0.3,   # auxiliary head
                "softmax_2": 1.0,   # main head
            },
            metrics={
                "softmax_1": ["accuracy"],
                "softmax_2": ["accuracy"],
            }
        )

        train_ds2 = train_ds.map(lambda x, y: (x, {"softmax_1": y, "softmax_2": y}))
        val_ds2   = val_ds.map(lambda x, y: (x, {"softmax_1": y, "softmax_2": y}))
        test_ds2  = test_ds.map(lambda x, y: (x, {"softmax_1": y, "softmax_2": y}))

        history = model_inception.fit(
            train_ds2,
            validation_data=val_ds2,
            epochs=10,
        )

        # For final prediction
        # pred1, pred2 = model_inception.predict(x_batch)
        # yhat = tf.argmax(pred2, axis=-1)  # final prediction from main head
        # OR
        # infer_model = tf.keras.Model(
        #     inputs=model_inception.input,
        #     outputs=model_inception.get_layer("softmax_2").output,
        #     name="cnn_inception_infer"
        # )
        # pred = infer_model.predict(x_batch)
        # yhat = tf.argmax(pred, axis=-1)
        




