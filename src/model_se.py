import tensorflow as tf
from tensorflow.keras import layers, models


def se_block(x, reduction=8, name="se"):
    """
    Squeeze-and-Excitation (SE) channel attention block.

    Args:
        x: 4D tensor (B, T, F, C)
        reduction: channel reduction ratio (e.g., 8 or 16)
        name: base name for layers

    Returns:
        Tensor with channel-wise gating applied.
    """
    ch = x.shape[-1]
    if ch is None:
        raise ValueError("Channel dimension must be defined for SE block.")

    # Squeeze: global average pooling over (T, F) -> (B, C)
    s = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)

    # Excitation: small MLP -> (B, C)
    hidden = max(1, int(ch) // int(reduction))
    s = layers.Dense(hidden, activation="relu", name=f"{name}_fc1")(s)
    s = layers.Dense(int(ch), activation="sigmoid", name=f"{name}_fc2")(s)

    # Reshape to (B, 1, 1, C) and scale
    s = layers.Reshape((1, 1, int(ch)), name=f"{name}_reshape")(s)
    out = layers.Multiply(name=f"{name}_scale")([x, s])
    return out


def build_cnn_tpool2_se(input_shape=(32, 40, 1), num_classes=12, se_reduction=8):
    """
    Sainath-style-ish baseline + SE blocks.
    (Conv -> freq pooling -> Conv -> freq pooling -> ... -> head)
    Adjust this to match your exact cnn_trad_fpool3 if needed.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(94, kernel_size=(21, 8), strides=(1,1), padding="valid", activation="relu", name="conv1")(inputs)
    x = se_block(x, reduction=se_reduction, name="se1")
    x = layers.MaxPool2D(pool_size=(2, 3), strides=(2, 3), padding="valid", name="pool1_freq")(x)

    x = layers.Conv2D(94, kernel_size=(6, 4), strides=(1, 1), padding="valid", activation="relu", name="conv2")(x)
    x = se_block(x, reduction=se_reduction, name="se2")

    x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Dense(32, activation="relu", name="dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_tpool2_se")
    return model