import pathlib 
import platform
import tensorflow as tf

system = platform.system()

# Function to create file paths and labels for each split (train/test/valid)

def load_audio_dataset(data_dir, validation_samples, test_samples):

    data_dir = pathlib.Path(data_dir)

    # Read validation and test file lists
    with open(validation_samples, 'r') as f:
        val_files = set(line.strip() for line in f)

    with open(test_samples, 'r') as f:
        test_files = set(line.strip() for line in f)

    # Get class folders except background noise
    class_names = sorted([
        item.name for item in data_dir.glob("*/")
        if item.name not in ["_background_noise_", ".DS_Store", "LICENSE", "README.md", "testing_list.txt", "validation_list.txt"]
    ])

    class_index = {cls: i for i, cls in enumerate(class_names)}

    # Split lists
    train_files, train_labels, train_labels_str = [], [], []
    val_files_list, val_labels, val_labels_str = [], [], []
    test_files_list, test_labels, test_labels_str = [], [], []

    for cls in class_names:
        class_dir = data_dir / cls

        for audio_file in class_dir.glob("*.wav"):
            rel_path = f"{cls}/{audio_file.name}"

            # Assign split purely based on val/test lists
            if rel_path in test_files:
                test_files_list.append(str(audio_file))
                test_labels.append(class_index[cls])
                test_labels_str.append(cls)

            elif rel_path in val_files:
                val_files_list.append(str(audio_file))
                val_labels.append(class_index[cls])
                val_labels_str.append(cls)

            else:
                train_files.append(str(audio_file))
                train_labels.append(class_index[cls])
                train_labels_str.append(cls)

    return (
        train_files, train_labels, train_labels_str,
        val_files_list, val_labels, val_labels_str,
        test_files_list, test_labels, test_labels_str,
        class_names
    )


def apply_time_shift(wav, sample_rate, max_time_shift_ms):
    """
    Apply random time shift to the waveform.
    
    Args:
        wav: Input waveform tensor
        sample_rate: Sample rate (typically 16000)
        max_time_shift_ms: Maximum shift in milliseconds (set to 100 ms in our case, to mimic Tang paper)
    
    Returns:
        Time-shifted waveform
    """
    # Convert milliseconds to samples
    max_shift_samples = int((max_time_shift_ms / 1000.0) * sample_rate)
    
    # Generate random shift between -max_shift_samples and +max_shift_samples
    shift_samples = tf.random.uniform(
        shape=[],
        minval=-max_shift_samples,
        maxval=max_shift_samples + 1,
        dtype=tf.int32
    )

    # Get the length of the waveform
    wav_length = tf.shape(wav)[0]
    
    # Apply the shift using tf.roll (circular shift)
    shifted_wav = tf.roll(wav, shift=shift_samples, axis=0)
    
    return shifted_wav