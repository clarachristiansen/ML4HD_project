import tensorflow as tf
from tensorflow.keras import layers, models
from .preprocessing import get_datasets
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def build_cnn_autoencoder_for_svm(input_shape=(98, 40, 1), latent_dim=128):
    """
    CNN Autoencoder:
      - Full model to train: inputs -> recon
      - Encoder + bottleneck to use for SVM: inputs -> latent feature vector

    Returns:
      autoencoder: Model(inputs -> recon)
      encoder_for_svm: Model(inputs -> latent_vec)
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder -> bottleneck map
    y = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="enc_conv1")(inputs)
    y = layers.MaxPool2D((2, 2), padding="same", name="enc_pool1")(y)

    y = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="enc_conv2")(y)
    y = layers.MaxPool2D((2, 2), padding="same", name="enc_pool2")(y)

    y = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="enc_conv3")(y)
    y = layers.MaxPool2D((2, 2), padding="same", name="enc_pool3")(y)

    # bottleneck map (compressed channels)
    bottleneck_map = layers.Conv2D(64, (1, 1), padding="same", activation="relu",
                                   name="bottleneck_1x1")(y)

    # Decoder (upsample back)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="dec_conv1")(bottleneck_map)
    x = layers.UpSampling2D((2, 2), name="dec_up1")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="dec_conv2")(x)
    x = layers.UpSampling2D((2, 2), name="dec_up2")(x)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="dec_conv3")(x)
    x = layers.UpSampling2D((2, 2), name="dec_up3")(x)

    x = layers.Cropping2D(cropping=((3, 3), (0, 0)), name="crop_to_input")(x)

    # reconstruction (linear is fine if inputs are standardized; use sigmoid only if inputs in [0,1])
    recon = layers.Conv2D(1, (3, 3), padding="same", activation="linear", name="recon")(x)

    autoencoder = models.Model(inputs=inputs, outputs=recon, name="autoencoder")

    # Features for SVM (encoder + bottleneck only)
    feat = layers.GlobalAveragePooling2D(name="feat_gap")(bottleneck_map)
    latent_vec = layers.Dense(latent_dim, activation=None, name="feat_latent")(feat)
    encoder_for_svm = models.Model(inputs=inputs, outputs=latent_vec, name="encoder_for_svm")

    return autoencoder, encoder_for_svm

def extract_features(encoder_for_svm, ds, max_batches=None):
    """
    Extract (X, y) from a tf.data.Dataset using a trained encoder_for_svm.

    Assumes ds yields: (x_batch, y_batch)
      - x_batch shape: (B, T, F, 1)
      - y_batch shape: (B,) integer labels

    max_batches: optionally stop early for quick debugging.
    """
    X_list, y_list = [], []

    for i, (x, y) in enumerate(ds):
        if max_batches is not None and i >= max_batches:
            break

        # Ensure we're in inference mode for layers like Dropout/BN if present
        feats = encoder_for_svm(x, training=False).numpy()  # (B, latent_dim)
        X_list.append(feats)
        y_list.append(y.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

def build_svm(kernel="linear", C=1.0, gamma="scale", class_weight="balanced", probability=False):
    """
    Build an SVM classifier for the autoencoder features.

    Recommended starting point for many KWS setups:
      - kernel="linear" (fast, often strong)
      - class_weight="balanced" if classes are imbalanced

    Returns: sklearn Pipeline = StandardScaler -> SVM
    """
    if kernel == "linear":
        # Option A: SVC(kernel="linear") supports probability=True (slower)
        svm = LinearSVC(C=C, class_weight=class_weight, max_iter=20000)
        # Option B: LinearSVC is faster for large datasets (no probability)
        # svm = LinearSVC(C=C, class_weight=class_weight, max_iter=20000)
    else:
        svm = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight, probability=probability)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm),
    ])
    return model



if __name__ == "__main__":
    frames = 98
    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False, frames=frames)

    autoencoder, encoder_for_svm = build_cnn_autoencoder_for_svm()
    autoencoder.summary()
    encoder_for_svm.summary()

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    train_ae = train_ds.map(lambda x, y: (x, x))
    val_ae = val_ds.map(lambda x, y: (x, x))


    history = autoencoder.fit(
        train_ae,
        validation_data=val_ae,
        epochs=20,
    )
    print("Training of autoencoder complete.")

    X_train, y_train = extract_features(encoder_for_svm, train_ds)
    X_val, y_val     = extract_features(encoder_for_svm, val_ds)
    X_test, y_test   = extract_features(encoder_for_svm, test_ds)

    print(f"Extracted features shape: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    svm_model = build_svm(kernel="linear", C=1.0, class_weight="balanced", probability=False)
    svm_model.fit(X_train, y_train)

    # -------------------------
    # Evaluate
    # -------------------------
    val_pred = svm_model.predict(X_val)
    test_pred = svm_model.predict(X_test)

    print("\nValidation accuracy:", accuracy_score(y_val, val_pred))
    print("Test accuracy:", accuracy_score(y_test, test_pred))

    # Optional: detailed report
    target_names = classes  # assuming classes is list of class names aligned with label ids
    print("\nClassification report (test):")
    print(classification_report(y_test, test_pred, target_names=target_names, digits=3))

    # Optional: confusion matrix (numbers)
    cm = confusion_matrix(y_test, test_pred)
    print("\nConfusion matrix (test):")
    print(cm)