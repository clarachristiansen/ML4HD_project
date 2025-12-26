import tensorflow as tf
import numpy as np
import random
from pathlib import Path
import os
import pandas as pd

from .utils_lab import *
from .utils_preprocessing import load_data_from_files, create_tf_dataset

### GLOBAL variables
from config import (SAMPLE_RATE,
                    BATCH_SIZE,
                    N_MELS)

### Seed => reproduceability
seed = 24
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def get_datasets(sample_rate: int = SAMPLE_RATE, batch_size: int = BATCH_SIZE, repeat_train: bool = True, frames: int = 32) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str], list[str]]:
    """Build train/val/test datasets and return them plus metadata."""
    train_df, val_df, test_df, classes, background_noise_files = load_data_from_files()

    train_ds = create_tf_dataset(
        train_df,
        sample_rate=sample_rate,
        background_noise_files=background_noise_files,
        noise_prob=0.8,
        cache_file="training_set",
        batch_size=batch_size,
        shuffle=True,
        noise=True,
        repeat=repeat_train,
        final_data='logmel_spectrogram',
        num_mel_filters=N_MELS,
        train=True,
        frames=frames
    )

    valid_ds = create_tf_dataset(
        val_df,
        sample_rate=sample_rate,
        background_noise_files=background_noise_files,
        noise_prob=0.8,
        cache_file="validation_set",
        batch_size=batch_size,
        cache=False,
        final_data='logmel_spectrogram',
        num_mel_filters=N_MELS,
        frames=frames
    )

    test_ds = create_tf_dataset(
        test_df,
        sample_rate=sample_rate,
        background_noise_files=background_noise_files,
        noise_prob=0.8,
        batch_size=batch_size,
        final_data='logmel_spectrogram',
        num_mel_filters=N_MELS,
        frames=frames
    )

    return train_ds, valid_ds, test_ds, classes, background_noise_files


if __name__ == "__main__":
    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets()
    print(f"Number of classes: {len(classes)}  {classes}")
    batch = next(iter(train_ds))
    print(type(batch[0]), batch[0].shape, batch[1].shape)
    print("Datasets loaded successfully.")
    print("-"*30)