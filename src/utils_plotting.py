import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter





def plot_recording_waveform(audio_sample, noisy_audio_sample, sample_rate):
    """
    Plots the audio_sample and the noisy_audio_sample side by side.
    Assumes inputs are numpy arrays (samples) and already squeezed.
    """

    # 2. Create Time Axes
    # No TF checks performed here; assumes input is already numpy array
    time_axis_sample = np.linspace(0, len(audio_sample) / sample_rate, num=len(audio_sample))
    time_axis_noisy = np.linspace(0, len(noisy_audio_sample) / sample_rate, num=len(noisy_audio_sample))

    # 3. Plotting Side by Side
    plt.figure(figsize=(16, 6))

    # -- Plot 1: Original Audio Sample --
    plt.subplot(1, 2, 1)
    plt.plot(time_axis_sample, audio_sample, color='#1f77b4', alpha=0.8)
    plt.title("Original Audio Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    # -- Plot 2: Noisy Audio Sample --
    plt.subplot(1, 2, 2)
    plt.plot(time_axis_noisy, noisy_audio_sample, color='#ff7f0e', alpha=0.8)
    plt.title("Noisy/Shifted Audio Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#NOTE: function to be added to import, should be removed from the notebook




def plot_spectrogram_and_mfccs(spectrogram, mfccs):
    """
    Plots the Spectrogram and MFCCs side by side.
    
    Args:
        spectrogram: Tensor or array of shape (num_frames, num_freq_bins)
        mfccs: Tensor or array of shape (num_frames, num_coefficients)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Spectrogram ---
    # We take the Transpose (.T) so Time is on X-axis and Freq is on Y-axis
    # We also use log(spectrogram) for visualization because raw magnitudes 
    # have a high dynamic range and are hard to see.
    log_spec_viz = tf.math.log(spectrogram + np.finfo(float).eps).numpy().T
    
    im1 = axes[0].imshow(log_spec_viz, origin='lower', aspect='auto', cmap='viridis')
    axes[0].set_title('Log Spectrogram')
    axes[0].set_ylabel('Frequency Bin')
    axes[0].set_xlabel('Time (Frames)')
    fig.colorbar(im1, ax=axes[0], format='%+2.0f Log Power')

    # --- Plot 2: MFCCs ---
    # Transpose (.T) so Time is on X-axis and MFCC Coeffs are on Y-axis
    mfcc_viz = mfccs.numpy().T
    
    im2 = axes[1].imshow(mfcc_viz, origin='lower', aspect='auto', cmap='jet')
    axes[1].set_title('MFCCs (Static + Delta + Delta-Delta)')
    axes[1].set_ylabel('MFCC Coefficients')
    axes[1].set_xlabel('Time (Frames)')
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()





def plot_prediction_distribution(true_labels, pred_labels, classes):
  
    if hasattr(pred_labels, 'numpy'):
        pred_labels = pred_labels.numpy()
    if hasattr(true_labels, 'numpy'):
        true_labels = true_labels.numpy()


    for true_idx, true_name in enumerate(classes):

        # Find all indices where the True Label is the current class
        indices_of_current_class = np.where(true_labels == true_idx)[0]
        total_occurrences = len(indices_of_current_class)

        if total_occurrences == 0:
            continue # Skip if this word isn't in the test set

        # Get the model's predictions for these specific indices
        predictions_for_this_class = pred_labels[indices_of_current_class]

        # Count frequency of all predictions
        counts = Counter(predictions_for_this_class)

        # Get the Top 4 most frequent predictions
        top_k_predictions = counts.most_common(4)

        
        plot_labels = []
        plot_values = []
        bar_colors = []

        for pred_idx, count in top_k_predictions:
            pred_name = classes[pred_idx]
            plot_labels.append(pred_name)

            # Calculate proportion
            plot_values.append(count / total_occurrences)


            if pred_idx == true_idx:
                bar_colors.append('green')
            else:
                bar_colors.append('#d62728')

        # Plotting
        plt.figure(figsize=(8, 3))

        bars = plt.barh(range(len(plot_values)), plot_values, color=bar_colors)

        
        plt.yticks(range(len(plot_values)), plot_labels)
        plt.xlabel('Proportion of Total Instances')

        # Calculate recall
        correct_count = counts[true_idx]
        recall = correct_count / total_occurrences

        plt.title(f"True Label: '{true_name}' (Recall: {recall:.1%})\nDistribution of Predictions:")

        plt.xlim(0, 1.1) # Max 100% + padding
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        # Add text labels to bars
        for i, v in enumerate(plot_values):
            plt.text(v, i, f" {v*100:.1f}%", va='center', fontweight='bold', color='black')

        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()