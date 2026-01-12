import tensorflow as tf
from tensorflow.keras import layers, models


def cbam_block(x, reduction=8, spatial_kernel=7, name="cbam"):
    """
    CBAM block for 2D feature maps (B, T, F, C):
      1) Channel attention (avg+max pooling over spatial)
      2) Spatial attention (avg+max pooling over channels)

    Args:
        x: input tensor (B, T, F, C)
        reduction: channel reduction ratio
        spatial_kernel: kernel size for spatial attention conv (typically 7)
        name: base layer name

    Returns:
        Tensor after CBAM refinement.
    """
    ch = x.shape[-1]
    if ch is None:
        raise ValueError("Channel dimension must be defined for CBAM.")

    # -----------------------
    # 1) Channel attention
    # -----------------------
    avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_ch_gap")(x)  # (B, C)
    max_pool = layers.GlobalMaxPooling2D(name=f"{name}_ch_gmp")(x)      # (B, C)

    hidden = max(1, int(ch) // int(reduction))

    # Shared MLP (Dense -> Dense) applied to both pooled descriptors
    mlp1 = layers.Dense(hidden, activation="relu", name=f"{name}_ch_fc1")
    mlp2 = layers.Dense(int(ch), activation=None, name=f"{name}_ch_fc2")

    avg_out = mlp2(mlp1(avg_pool))
    max_out = mlp2(mlp1(max_pool))

    ch_attn = layers.Add(name=f"{name}_ch_add")([avg_out, max_out])
    ch_attn = layers.Activation("sigmoid", name=f"{name}_ch_sigmoid")(ch_attn)
    ch_attn = layers.Reshape((1, 1, int(ch)), name=f"{name}_ch_reshape")(ch_attn)

    x = layers.Multiply(name=f"{name}_ch_scale")([x, ch_attn])

    # -----------------------
    # 2) Spatial attention
    # -----------------------
    # Pool along channels -> (B, T, F, 1) for avg and max
    avg_map = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                            name=f"{name}_sp_avg")(x)
    max_map = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                            name=f"{name}_sp_max")(x)

    sp = layers.Concatenate(axis=-1, name=f"{name}_sp_concat")([avg_map, max_map])  # (B, T, F, 2)

    sp_attn = layers.Conv2D(
        filters=1,
        kernel_size=(spatial_kernel, spatial_kernel),
        strides=(1, 1),
        padding="same",
        activation="sigmoid",
        name=f"{name}_sp_conv",
    )(sp)

    out = layers.Multiply(name=f"{name}_sp_scale")([x, sp_attn])
    return out


def build_cnn_tpool2_cbam(input_shape=(32, 40, 1), num_classes=12, cbam_reduction=8, spatial_kernel=7):
    """
    Your tpool2-style CNN + CBAM blocks.
    CBAM inserted after each Conv2D.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(94, kernel_size=(21, 8), strides=(1, 1), padding="valid",
                      activation="relu", name="conv1")(inputs)
    x = cbam_block(x, reduction=cbam_reduction, spatial_kernel=spatial_kernel, name="cbam1")
    x = layers.MaxPool2D(pool_size=(2, 3), strides=(2, 3), padding="valid", name="pool1")(x)

    x = layers.Conv2D(94, kernel_size=(6, 4), strides=(1, 1), padding="valid",
                      activation="relu", name="conv2")(x)
    x = cbam_block(x, reduction=cbam_reduction, spatial_kernel=spatial_kernel, name="cbam2")

    x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Dense(32, activation="relu", name="dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_tpool2_cbam")
    return model
