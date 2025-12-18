import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tensorflow_gnn as tfgnn
from pathlib import Path
from .utils_lab import load_audio_dataset
from typing import Literal


### GLOBAL variables
from config import (FRAME_LENGTH,
                    FRAME_STEP,
                    SAMPLE_RATE,
                    N_DILATION_LAYERS,
                    REDUCED_NODE_REP_BOOL,
                    REDUCED_NODE_REP_K,
                    WINDOW_SIZE_SIMPLE,
                    N_FRAMES,
                    SCRIPT_DIR)

### Prepare waveform functions
def read_path_to_wav(file_path):
    file_contents = tf.io.read_file(file_path)
    wav, _ = tf.audio.decode_wav(file_contents)
    return wav

def adjust_audio_length(wav, sample_rate):
  target_length = sample_rate
  current_length = tf.shape(wav)[0]
  if current_length > target_length:
      wav = wav[:target_length]
  else:
      paddings = [[0, target_length - current_length], [0, 0]]
      wav = tf.pad(wav, paddings)
  return tf.squeeze(wav, axis=-1)

def apply_time_shift(wav, sample_rate, max_time_shift_ms = 100):
    max_shift_samples = int((max_time_shift_ms / 1000.0) * sample_rate)
    shift_samples = tf.random.uniform(shape=[], minval=-max_shift_samples, maxval=max_shift_samples + 1, dtype=tf.int32)
    shifted_audio = tf.roll(wav, shift=shift_samples, axis=0)
    return shifted_audio

def apply_random_noise(wav, background_noise_files, target_length, noise_prob, min_snr_db=0, max_snr_db=10):

    background_noise_files = tf.convert_to_tensor(background_noise_files, dtype=tf.string)

    # Select a random noise file
    noise_file = tf.random.shuffle(background_noise_files)[0]
    noise_wav = read_path_to_wav(noise_file)

    # Extract a random segment of the desired length
    noise_len = tf.shape(noise_wav)[0]
    max_start = tf.maximum(noise_len - target_length, 1)
    start = tf.random.uniform([], 0, max_start, dtype=tf.int32)
    noise = noise_wav[start : start + target_length]

    noise = tf.squeeze(noise, axis=-1) if tf.rank(noise) > 1 else noise

    # Compute scaling for SNR
    signal_power = tf.reduce_mean(tf.square(wav))
    noise_power = tf.reduce_mean(tf.square(noise))

    snr_db = tf.random.uniform([], min_snr_db, max_snr_db)
    snr_lin = tf.pow(10.0, snr_db / 10.0)

    scale = tf.sqrt(signal_power / (noise_power * snr_lin + 1e-12))
    noise = noise * scale

    # Apply noise with given probability
    do_apply = tf.random.uniform([]) <= noise_prob
    noise_signal = tf.where(do_apply, wav + noise, wav)
    return noise_signal


# Convert waveform to spectrogram
def get_spectrogram(wav, sample_rate):

    frame_length = int(sample_rate * 0.025)
    frame_step = int(sample_rate * 0.010)

    spectrogram = tf.signal.stft(wav,
                                 frame_length= frame_length,
                                 frame_step= frame_step,
                                 window_fn= tf.signal.hamming_window)

    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)

    return spectrogram, wav

# Compute Mel-filterbank features and MFCCs
def apply_mel_filterbanks(spectrogram, wav, sample_rate, num_mel_filters=13):

    # Obtain the number of frequency bins of our spectrogram.
    num_spectrogram_bins = tf.shape(spectrogram)[-1]

    # Define the frequency band we are intereted in
    min_frequency = 100
    max_frequency = sample_rate/2

    # Create Mel filterbank
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(num_mel_filters, num_spectrogram_bins, sample_rate, min_frequency, max_frequency)

    # Apply the transformation
    mel_spectrogram = tf.tensordot(spectrogram, mel_filterbank, axes= 1)

    # Set output shape
    output_shape = tf.concat([tf.shape(spectrogram)[:-1], [tf.shape(mel_filterbank)[-1]]], axis=0)
    mel_spectrogram = tf.reshape(mel_spectrogram, output_shape)

    # Compute a stabilized log to get log-magnitude mel-scale spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + np.finfo(float).eps)

    return log_mel_spectrogram, wav


def compute_delta(mfccs, M):

        frame_count = tf.shape(mfccs)[0]
        # Pad the mfccs at the beginning and at the end to handle boundary frames
        padded_mfccs = tf.pad(mfccs, [[M, M], [0, 0]], mode='SYMMETRIC')    # This pads [M,M] in time (frames) dimension and pads [0,0] in the frequency dimension

        denominator = 2 * sum([m**2 for m in range(1, M+1)])
        # Initialize the deltas
        deltas = tf.zeros_like(mfccs)

        for m in range(1, M+1):
            # Get frames at n+m
            next_frames = padded_mfccs[M + m: M + m + frame_count]
            # Get frames at n-m
            prev_frames = padded_mfccs[M - m : M - m + frame_count]
            # Add weighted difference to the delta coefficients
            deltas += m * (next_frames - prev_frames) / denominator

        return deltas

def get_mfccs(log_mel_spectrogram, wav, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, M = 2):

    # compute DCT and select coefficients, discarding the first
    mfccs_0 = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., 1:]

    # compute the first derivative of the MFCCs
    mfccs_delta_1 = compute_delta(mfccs_0, M)

    # compute the second derivative of the MFCCs
    mfccs_delta_2 = compute_delta(mfccs_delta_1, M)

    # Divide the raw audio into frames
    framed_wav = tf.signal.frame(wav, frame_length= frame_length, frame_step= frame_step)

    # Compute the log energy of each frame
    frame_energy = tf.reduce_sum(framed_wav**2, axis=-1)
    log_frame_energy = tf.math.log(frame_energy + np.finfo(float).eps)/tf.math.log(10.0)
    log_frame_energy = tf.expand_dims(log_frame_energy, axis=-1)

    # Compute second energy
    energy_delta_1 = compute_delta(log_frame_energy, M)

    # Compute third energy
    energy_delta_2 = compute_delta(energy_delta_1, M)

    # obtain final MFCCs
    mfccs = tf.concat([mfccs_0, log_frame_energy, mfccs_delta_1, energy_delta_1, mfccs_delta_2, energy_delta_2], axis=-1)

    return mfccs

# Adjacency matrix creation functions
def create_dilated_adjacency_matrix(adjacency_matrix, dilation_rate):

    powers = [adjacency_matrix] # where to store all matrix prowers, starting from powers[0] = A^1

    A_power = adjacency_matrix

    for k in range(1, dilation_rate):
        A_power = tf.linalg.matmul(A_power, adjacency_matrix)
        powers.append(A_power)

    A_d = powers[-1]

    # remove all lower-hop connections: A^1, A^2, ..., A^(d-1)
    for p in powers[:-1]:
        A_d = tf.where(p > 0, tf.zeros_like(A_d), A_d)

    # Remove self-loops from the matrix setting the diagonal to zeros
    A_d = tf.linalg.set_diag(A_d, tf.zeros(tf.shape(A_d)[0], dtype=tf.float32))

    # binarize the adj matrix (keep all entries > 0)
    A_d = tf.cast(A_d > 0, tf.float32)

    return A_d

def create_adjacency_matrix(mfcc, num_frames, label, n_dilation_layers=0, window_size=5):

    adjacency_matrices = []
    # Students compute the pairwise distances between frame indices
    indices = tf.range(num_frames, dtype=tf.int32)
    i = tf.reshape(indices, [-1, 1])
    j = tf.reshape(indices, [1, -1])

    distance = tf.abs(i - j)

    # create adjacency by thresholding with window_size
    adjacency_matrix = tf.cast(distance <= window_size, tf.float32)

    # Remove self-loops
    adjacency_matrix = tf.linalg.set_diag(
            adjacency_matrix,
            tf.zeros(num_frames, dtype=tf.float32)
        )

    adjacency_matrices.append(adjacency_matrix)

    dilation_rate = 2

    for layer in range(n_dilation_layers):

        dilated_A = create_dilated_adjacency_matrix(adjacency_matrix,
                                                         dilation_rate=dilation_rate)

        adjacency_matrices.append(dilated_A)

        dilation_rate += 2  # Increase for next layer

    return adjacency_matrices


# Convert MFCCs to graph tensors
def mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrices, label, reduced_node_bool, reduced_node_k):

    # reduce the node representation
    if reduced_node_bool:
        if ((98 // reduced_node_k) == (98/ reduced_node_k)):
          mfcc_static = tf.reshape(mfcc, [98 // reduced_node_k, 39])
        else:
          # this case when we have some overhang of a group
          mfcc_static = tf.reshape(mfcc, [((98 // reduced_node_k) + 1), 39])
    else:
        mfcc_static = tf.reshape(mfcc, [98, 39])

    # Create the node set
    node_sets = {
              "frames": tfgnn.NodeSet.from_fields(features={"features": mfcc_static}, sizes=[tf.shape(mfcc_static)[0]])
                }
    
    # Create an edge set for each adjacency matrix
    edge_sets = {}

    # Unstack the matrices so we can iterate over them
    unstacked_matrices = tf.unstack(adjacency_matrices, axis=0)

    for i, adjacency_matrix in enumerate(unstacked_matrices):
        # Get edges from this adjacency matrix
        edges = tf.where(adjacency_matrix > 0)

        # Get corresponding weights
        weights = tf.gather_nd(adjacency_matrix, edges)

        # Extract source and target indices
        sources = edges[:, 0]
        targets = edges[:, 1]

        # Create edge set with unique names
        edge_set_name = f"connections_{i}"
        edge_sets[edge_set_name] = tfgnn.EdgeSet.from_fields(
            features={"weights" : weights},
            sizes=[tf.shape(edges)[0]],
            adjacency=tfgnn.Adjacency.from_indices(
                source=("frames", sources),
                target=("frames", targets)
            )
        )

    # Create the graph tensor with all node sets and edge sets
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets,
        edge_sets=edge_sets
    )

    return graph_tensor

def prepare_mel_for_cnn(mel_spec, wav, label, desired_frames=32):
    """
    mel_spec: [T, n_mels]
    wav: unused here, just passed through by pipeline
    label: scalar
    Returns:
        mel_img: [desired_frames, n_mels, 1]
        label: scalar
    """
    num_frames = tf.shape(mel_spec)[0]
    print('Desired frames:', desired_frames)
    def crop():
        if True:
            start = tf.random.uniform([], 0, num_frames - desired_frames + 1, dtype=tf.int32)
            return mel_spec[start:start + desired_frames, :]
        else:
            return mel_spec[:desired_frames, :]

    def pad():
        pad_len = desired_frames - num_frames
        return tf.pad(mel_spec, [[0, pad_len], [0, 0]])

    mel_spec = tf.cond(num_frames >= desired_frames, crop, pad)

    # Add channel dimension for Conv2D: [T, F, 1]
    mel_spec = tf.expand_dims(mel_spec, -1)

    return mel_spec, label

def assert_finite_2(x, y):
    tf.debugging.assert_all_finite(x, "NaN/Inf 2,1 HERE")
    return x, y

def assert_finite_3(x, y, z):
    tf.debugging.assert_all_finite(x, "NaN/Inf 3,1 HERE")
    tf.debugging.assert_all_finite(y, "NaN/Inf 3,2 HERE")
    return x, y, z

# Final dataset creation function
def create_tf_dataset(dataframe, sample_rate, background_noise_files, noise_prob, cache_file = '', batch_size = 16, cache = False, shuffle = False, repeat = False, noise = False, final_data = Literal['logmel_spectrogram','graph_tensor'], num_mel_filters=13) -> tf.data.Dataset:

    # obtain file names and numerical labels
    file_names, labels = dataframe["file_path"], tf.cast(dataframe["label"], tf.int32)

    # Create the Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))

    if shuffle:
        dataset = dataset.shuffle(len(file_names), reshuffle_each_iteration=True)

    # Loading and decoding wav files using read_path_to_wav and then map the function to the dataset
    read_path_to_wav_lambda = lambda file_path, label: (read_path_to_wav(file_path), label)

    dataset = dataset.map(read_path_to_wav_lambda, num_parallel_calls = os.cpu_count())

    # Adjust audio length using adjust_audio_length and then map the function to the dataset
    adjust_audio_length_lambda = lambda wav, label: (adjust_audio_length(wav, sample_rate), label)
    dataset = dataset.map(adjust_audio_length_lambda, num_parallel_calls = os.cpu_count())

    # Caching the dataset after deterministic transformation (Optional)
    # if cache:
    #   dataset = dataset.cache(cache_file).shuffle(1000)

    if noise:
        #Applying time-shift using apply_time_shift() and then map the function to the dataset
        apply_time_shift_lambda = lambda wav, label: (apply_time_shift(wav, sample_rate), label)
        dataset = dataset.map(apply_time_shift_lambda, num_parallel_calls=os.cpu_count())

        #Applying background noise using apply_random_noise() and then map the function to the dataset
        apply_random_noise_lambda = lambda wav, label: (apply_random_noise(wav, background_noise_files, sample_rate, noise_prob), label)
        dataset = dataset.map(apply_random_noise_lambda, num_parallel_calls=os.cpu_count())

    # Compute the spectrogram using get_spectrogram
    # use *get_spectrogram(wav, sample_rate) as the function returns a tuple of elements - more compact version of get_spectrogram(wav, sample_rate)[0], get_spectrogram(wav, sample_rate)[1]
    get_spectrogram_lambda = lambda wav, label: (*get_spectrogram(wav, sample_rate), label)
    dataset = dataset.map(get_spectrogram_lambda, num_parallel_calls=os.cpu_count())


    # Apply the Mel filters
    # use *apply_mel_filterbanks(spec, wav, sample_rate) - same as above
    apply_mel_filterbanks_lambda = lambda spec, wav, label: (*apply_mel_filterbanks(spec, wav, sample_rate, num_mel_filters=num_mel_filters), label)
    dataset = dataset.map(apply_mel_filterbanks_lambda, num_parallel_calls = os.cpu_count())
    # Check for NaNs right after mel stage
    ds_mid = dataset.take(1)  # right after mel stage in your code
    mel_spec, wav, label = next(iter(ds_mid))
    print("NaNs after mel:", tf.reduce_sum(tf.cast(tf.math.is_nan(mel_spec), tf.int32)).numpy(), flush=True)

    if final_data == 'graph_tensor':
        # Compute MFCC + Delta features using the get_mfccs() function
        get_mfccs_lambda = lambda spec, wav, label: (get_mfccs(spec, wav), label)
        dataset = dataset.map(get_mfccs_lambda, num_parallel_calls = os.cpu_count())

        # Create adjacency matrix using the create_adjacency_matrix() function
        create_adjacency_matrix_lambda = lambda mfcc, label: (mfcc ,create_adjacency_matrix(mfcc, N_FRAMES, label, n_dilation_layers= N_DILATION_LAYERS, window_size=  WINDOW_SIZE_SIMPLE), label)
        dataset = dataset.map(create_adjacency_matrix_lambda, num_parallel_calls = tf.data.AUTOTUNE)

        # Convert MFCCs to graph tensor using mfccs_to_graph_tensors_for_dataset() function
        mfccs_to_graph_tensors_for_dataset_lambda = lambda mfcc, adjacency_matrix, label: (mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrix, label, REDUCED_NODE_REP_BOOL, REDUCED_NODE_REP_K), label)
        dataset = dataset.map(mfccs_to_graph_tensors_for_dataset_lambda, num_parallel_calls = tf.data.AUTOTUNE)
        print('Converted MFCCs to Graph Tensors')
    elif final_data == 'logmel_spectrogram':
        
        # Compute Log-Mel Spectrograms using the get_log_mel_spectrogram() function
        dataset = dataset.map(
            lambda mel_spec, wav, label: prepare_mel_for_cnn(mel_spec, wav, label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds_end = dataset.take(1)
        x, y = next(iter(ds_end))
        print("NaNs after prepare:", tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int32)).numpy(), flush=True)
        print('Converted audio to Log-Mel Spectrograms')

    if cache:
        dataset = dataset.cache(cache_file)

    if repeat:
        dataset = dataset.repeat()
        print('Repeated dataset')

    dataset = dataset.batch(batch_size = batch_size)
    print('Batched dataset')

    dataset = dataset.prefetch(buffer_size = 1)
    print('Prefetched dataset', end = '\n-----\n')

    return dataset


def load_data_from_files(verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:
    """
    Load audio data from files. 

    Args:
        verbose (bool): If True, prints data summary. Default is False.

    Returns:
        train_df (pd.DataFrame): DataFrame containing training file_paths and labels.
        val_df (pd.DataFrame): DataFrame containing validation file_paths and labels.
        test_df (pd.DataFrame): DataFrame containing test file_paths and labels.
        classes (list): List of class labels.
        background_noise_files (list): List of background noise file paths.
    
    Raises:
        None
    """
    data_dir = SCRIPT_DIR.parent / 'data'
    validation_samples = 'validation_list.txt'
    test_samples = 'testing_list.txt'
    background_noise_dir = os.path.join(data_dir, '_background_noise_')
    background_noise_files = tf.io.gfile.glob(str(Path(background_noise_dir) / '*.wav'))

    train_files, train_labels, train_labels_str, val_files, val_labels, val_labels_str, test_files, test_labels, test_labels_str, classes = load_audio_dataset(
                data_dir=data_dir,
                validation_samples=os.path.join(data_dir, validation_samples),
                test_samples=os.path.join(data_dir, test_samples)
            )

    if verbose:
        print("Data summary:")
        total = len(train_files) + len(val_files) + len(test_files)
        print(f"Percentage of train samples: {len(train_files)/total*100:.1f}%")
        print(f"Percentage of validation samples: {len(val_files)/total*100:.1f}%")
        print(f"Percentage of test samples: {len(test_files)/total*100:.1f}%")
        print("\nTotal Number Samples :" , total)

    train_df = pd.DataFrame({
            'file_path': train_files,
            'label': train_labels,
            'label_str': train_labels_str,
            'split': 'train'})

    val_df = pd.DataFrame({
            'file_path': val_files,
            'label': val_labels,
            'label_str': val_labels_str,
            'split': 'val'})

    test_df = pd.DataFrame({
            'file_path': test_files,
            'label': test_labels,
            'label_str': test_labels_str,
            'split': 'test'})
    return train_df, val_df, test_df, classes, background_noise_files




if __name__ == "__main__":
    print("This is a utility module for preprocessing. Please import it to use its functions.")

