import tensorflow as tf
from tensorflow.keras import layers, models


def inception_block_scale_attention(inputs, filters, name="incept_attn", reduction=8):
    """
    Inception block with *scale attention* (branch weighting).
    Learns per-sample weights for the four branches and applies them before concatenation.

    Args:
        inputs: (B, T, F, C)
        filters: base branch width (int)
        reduction: bottleneck size for the attention MLP

    Returns:
        concatenated tensor after branch weighting
    """
    f = int(filters)
    f3_red = int(round(f * 1.5))
    f3_out = int(round(f * 2.0))
    f5_red = int(round(f * 0.5))
    f5_out = f
    fpool  = f
    f1_out = f

    # ----- branches -----
    b1 = layers.Conv2D(f1_out, (1, 1), padding="same", activation="relu",
                       name=f"{name}_1x1")(inputs)

    b2 = layers.Conv2D(f3_red, (1, 1), padding="same", activation="relu",
                       name=f"{name}_3x3_1x1")(inputs)
    b2 = layers.Conv2D(f3_out, (3, 3), padding="same", activation="relu",
                       name=f"{name}_3x3")(b2)

    b3 = layers.Conv2D(f5_red, (1, 1), padding="same", activation="relu",
                       name=f"{name}_5x5_1x1")(inputs)
    b3 = layers.Conv2D(f5_out, (5, 5), padding="same", activation="relu",
                       name=f"{name}_5x5")(b3)

    b4 = layers.MaxPool2D((3, 3), strides=(1, 1), padding="same",
                          name=f"{name}_pool")(inputs)
    b4 = layers.Conv2D(fpool, (1, 1), padding="same", activation="relu",
                       name=f"{name}_pool_1x1")(b4)

    branches = [b1, b2, b3, b4]

    # ----- scale attention (one scalar weight per branch per sample) -----
    # Get a descriptor per branch: (B, C_branch)
    descs = [layers.GlobalAveragePooling2D(name=f"{name}_gap_{i+1}")(b) for i, b in enumerate(branches)]

    # Reduce each descriptor to a scalar "score"
    # (keeps it small and makes it easy to compare branches of different channel sizes)
    scores = []
    for i, d in enumerate(descs):
        hidden = max(1, d.shape[-1] // reduction)
        s = layers.Dense(hidden, activation="relu", name=f"{name}_attn_fc1_{i+1}")(d)
        s = layers.Dense(1, activation=None, name=f"{name}_attn_fc2_{i+1}")(s)  # scalar
        scores.append(s)

    # Stack -> (B, 4) and softmax to get weights summing to 1
    score_vec = layers.Concatenate(axis=-1, name=f"{name}_attn_scores")(scores)  # (B, 4)
    alpha = layers.Activation("softmax", name=f"{name}_attn_softmax")(score_vec) # (B, 4)

    # Apply weights to branches
    weighted = []
    for i, b in enumerate(branches):
        # alpha[:, i] -> (B,), reshape -> (B,1,1,1), broadcast multiply
        a_i = layers.Lambda(lambda t, idx=i: tf.reshape(t[:, idx], (-1, 1, 1, 1)),
                            name=f"{name}_attn_reshape_{i+1}")(alpha)
        weighted.append(layers.Multiply(name=f"{name}_attn_mul_{i+1}")([b, a_i]))

    # concat weighted branches
    out = layers.Concatenate(axis=-1, name=f"{name}_concat")(
        weighted
    )
    return out


def build_cnn_inception_1_attention(input_shape=(32, 40, 1), num_classes=12):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), padding="valid", activation="relu", name="conv1")(inputs)
    x = layers.MaxPool2D((2, 2), strides=(2, 2), padding="valid", name="pool1")(x)

    x = layers.Conv2D(64, (1, 1), padding="valid", activation="relu", name="conv2_1x1")(x)
    x = layers.Conv2D(128, (3, 3), padding="valid", activation="relu", name="conv2_3x3")(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2), padding="valid", name="pool2")(x)

    # Inception with branch attention
    x = inception_block_scale_attention(x, filters=32, name="inception_1_attn", reduction=8)

    x = layers.AveragePooling2D((3, 3), strides=3, name="avgpool")(x)
    x = layers.Conv2D(128, (1, 1), padding="same", activation="relu", name="proj_1x1")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dnn_128")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_inception_1_attn")

    return model


def build_cnn_inception_2_attention(input_shape=(32, 40, 1), num_classes=12):
    inputs = layers.Input(shape=input_shape)

    # Conv1: m=7 (time), r=7 (freq), n=64, pool q=3 in frequency
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding="valid", activation="relu", name="conv1",)(inputs)
    # Max-pool pool_size=(2,2)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="pool1")(x)
    # Conv2: m=1(time), r=1 (freq), n=64, pool q=3 in frequency
    x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv2_1x1")(x)
    # Conv3: m=3(time), r=3 (freq), n=128, pool q=3 in frequency
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu", name="conv2_3x3")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="pool2")(x)
    x = inception_block_scale_attention(x, filters=32, name="inception_1_attn", reduction=8)

    x1 = layers.AveragePooling2D((3, 3), strides=3)(x)
    x1 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = layers.Flatten(name="flatten1")(x1)
    x1 = layers.Dense(128, activation="relu", name="dnn_128")(x1)
    x1 = layers.Dropout(0.5)(x1)
    # Softmax output
    outputs1 = layers.Dense(num_classes, activation="softmax", name="softmax_1")(x1)

    x = inception_block_scale_attention(x, filters=64, name="inception_2_attn", reduction=8)

    x2 = layers.AveragePooling2D((3, 3), strides=3)(x)
    x2 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x2)
    x2 = layers.Flatten(name="flatten2")(x2)
    x2 = layers.Dense(256, activation="relu", name="dnn_256")(x2)
    x2 = layers.Dropout(0.5)(x2)
    # Softmax output
    outputs2 = layers.Dense(num_classes, activation="softmax", name="softmax_2")(x2)

    model = models.Model(inputs=inputs, outputs=[outputs1, outputs2], name="cnn_inception_2_attn")

    return model


