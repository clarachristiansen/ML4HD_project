# src/demo.py
import argparse
import time
import numpy as np
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from .utils_preprocessing import (
    create_tf_dataset,
)

from .model_sainath import build_cnn_trad_fpool3, build_cnn_tpool2
from .model_se import build_cnn_tpool2_se
from .model_cbam import build_cnn_tpool2_cbam
from .model_inception import build_cnn_inception_1, build_cnn_inception_2
from .model_inception_attention import build_cnn_inception_1_attention, build_cnn_inception_2_attention

# Optional: read sample rate / n_mels from config if available
try:
    from config import SAMPLE_RATE, N_MELS
except Exception:
    SAMPLE_RATE = 16000
    N_MELS = 40


def parse_args():
    p = argparse.ArgumentParser(description="1-second KWS demo (record -> preprocess -> infer).")
    p.add_argument("--architecture", type=str, default="cnn_tpool2",
                   help="cnn_trad_fpool3 | cnn_tpool2 | cnn_tpool2_se | cnn_tpool2_cbam | "
                        "cnn_inception_1 | cnn_inception_1_attention | cnn_inception_2 | cnn_inception_2_attention")
    p.add_argument("--final_data", type=str, default="logmel_spectrogram",
                   help="logmel_spectrogram | logmel_spectrogram_bins | mel_pcen_a | mel_pcen_b | mfccs")
    p.add_argument("--frames", type=int, default=98)
    p.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--n_mels", type=int, default=N_MELS)
    p.add_argument("--seconds", type=float, default=1.0, help="Recording length in seconds (keep at 1.0).")
    p.add_argument("--use_wandb", action="store_true", default=True, help="Enable Weights & Biases logging.")
    p.add_argument("--out_wav", type=str, default="results/demo/demo.wav")


    # Checkpoint prefix (NO .index / .data suffix)
    p.add_argument("--ckpt_prefix", type=str,
                   default="results/results/checkpoint_e43lzt37.ckpt",
                   help="Path prefix to checkpoint, e.g. results/results/checkpoint_e43lzt37.ckpt")

    # Top-k printing
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


def build_model(architecture: str, num_classes: int, input_shape: tuple) -> tf.keras.Model:
    if architecture == "cnn_trad_fpool3":
        return build_cnn_trad_fpool3(num_classes=num_classes, input_shape=input_shape)
    if architecture == "cnn_tpool2":
        return build_cnn_tpool2(num_classes=num_classes, input_shape=input_shape)
    if architecture == "cnn_tpool2_se":
        return build_cnn_tpool2_se(num_classes=num_classes, input_shape=input_shape)
    if architecture == "cnn_tpool2_cbam":
        return build_cnn_tpool2_cbam(num_classes=num_classes, input_shape=input_shape)
    if architecture == "cnn_inception_1":
        return build_cnn_inception_1(num_classes=num_classes, input_shape=input_shape)
    if architecture == "cnn_inception_1_attention":
        return build_cnn_inception_1_attention(num_classes=num_classes, input_shape=input_shape)
    if architecture == "cnn_inception_2":
        return build_cnn_inception_2(num_classes=num_classes, input_shape=input_shape)
    if architecture == "cnn_inception_2_attention":
        return build_cnn_inception_2_attention(num_classes=num_classes, input_shape=input_shape)
    raise ValueError(f"Unknown architecture: {architecture}")

def make_demo_dataset(wav_path: str, final_data: str, frames: int, sample_rate: int, num_mel_filters: int):
    """
    Reuse your existing pipeline by creating a one-row DataFrame and calling create_tf_dataset.
    """
    df = pd.DataFrame({
        "file_path": [str(Path(wav_path).resolve())],
        "label": [0],            # dummy label; not used for inference
        "label_str": ["demo"],
        "split": ["test"],
    })

    # background_noise_files is only used if noise=True; keep noise=False here
    background_noise_files = []
    ds = create_tf_dataset(
        df,
        sample_rate=sample_rate,
        background_noise_files=background_noise_files,
        noise_prob=0.0,
        batch_size=1,
        shuffle=False,
        repeat=False,
        noise=False,
        train=False,
        final_data=final_data,
        num_mel_filters=num_mel_filters,
        frames=frames,
    )
    return ds

def record_1s_to_wav(path: str, sample_rate: int, seconds: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = int(sample_rate * seconds)

    print(f"Recording {seconds:.1f}s…")
    audio = sd.rec(frames=n, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio, axis=-1)

    # write float32 wav
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    print(f"Saved: {path}")

def main():
    args = parse_args()
    classes = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    num_classes = len(classes)

    # Input shape must match training setup
    if args.final_data == "mfccs":
        channels = 120  # your training script uses 120 for MFCCs
    else:
        channels = args.n_mels

    input_shape = (args.frames, channels, 1)

    # Build and load model
    model = build_model(args.architecture, num_classes=num_classes, input_shape=input_shape)
    model.load_weights("results/results/demo_model_incepattn_logmel.weights.h5")

    print("Checkpoint restore: OK")

    print("\nSay one of these classes:")
    for i, c in enumerate(classes):
        print(f"  {i:2d}: {c}")
    print("")

    print("Press ENTER to record 1 second. Type 'q' + ENTER to quit.\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd in {"q", "quit", "exit"}:
            break

        record_1s_to_wav(args.out_wav, args.sample_rate, args.seconds)

        ds = make_demo_dataset(
            wav_path=args.out_wav,
            final_data=args.final_data,
            frames=args.frames,
            sample_rate=args.sample_rate,
            num_mel_filters=N_MELS,
        )

        x, _ = next(iter(ds.take(1)))
        plt.figure(figsize=(10,10))
        plt.title("Demo Input Processed Feature")
        img = tf.squeeze(x).numpy()
        plt.imshow(img.T, aspect="auto", origin="lower", cmap="jet")
        plt.colorbar()
        plt.xlabel("Time (Frames)")
        plt.ylabel("Frequency Bins")
        plt.tight_layout()
        plt.savefig("results/demo_test.png", dpi=1000)
        plt.show()

        t0 = time.perf_counter()
        out = model(x, training=False)
        t1 = time.perf_counter()

        vals = out[0].numpy() 

        print("Output stats:",
            "min", float(vals.min()),
            "max", float(vals.max()),
            "sum", float(vals.sum()))

        
        probs = out.numpy().reshape(-1) 

        pred = int(np.argmax(probs))
        k = min(args.topk, len(classes))
        topk_idx = np.argsort(-probs)[:k]
        topk_str = ", ".join([f"{classes[i]} ({probs[i]:.2f})" for i in topk_idx])

        print(f"Pred: {classes[pred]} | p={probs[pred]:.2f} | infer={(t1-t0)*1000:.2f} ms")
        print(f"Top-{k}: {topk_str}\n")



if __name__ == "__main__":
    main()
