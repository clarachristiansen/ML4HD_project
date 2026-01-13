# evaluate.py
import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import wandb
from .utils_evaluate import compute_topk_accuracy, collect_predictions, load_weights, log_confusion_and_examples_spectrogram, log_confusion_and_examples_graph, extract_misclassified_graph_rows
from .preprocessing import get_datasets
from .model_sainath import build_cnn_trad_fpool3, build_cnn_tpool2
from .model_se import build_cnn_tpool2_se
from .model_cbam import build_cnn_tpool2_cbam
from .model_inception import build_cnn_inception_1, build_cnn_inception_2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ML4HD_project")
    p.add_argument("--wandb_entity", type=str, default="clara-christiansen-danmarks-tekniske-universitet-dtu")
    p.add_argument("--wandb_run_id", type=str, default=None, help="If set, resume/log into this exact run id")
    # Where to load best weights from
    p.add_argument("--download_dir", type=str, default=None, help="Local checkpoint prefix, e.g. checkpoints/best.ckpt")
    p.add_argument("--weights_path_wandb", type=str, default=None, help="W&B file name, e.g. model_best_weights_sainath")

    # Data / model configuration
    p.add_argument("--final_data", type=str, default='logmel_spectrogram', help="Type of input features: 'logmel_spectrogram', 'logmel_spectrogram_bins', 'mel_pcen_a', 'mel_pcen_b' or 'mfccs'")
    p.add_argument("--architecture", type=str, default="cnn_tpool2")
    p.add_argument("--frames", type=int, default=98)
    return p.parse_args()


def main():
    args = parse_args()
    print("Evaluation args:", args)

    if args.final_data == 'mfccs':
        channels = 120
        input_shape = (args.frames, channels, 1)
    else:
        channels = 40
        input_shape = (args.frames, channels, 1)
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            # If wandb_run_id is provided, this will attach to the SAME run
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            job_type="eval",
        )

    _, _, test_ds, classes, _ = get_datasets(repeat_train=False, frames=args.frames, final_data=args.final_data)
    class_names = list(classes) if isinstance(classes, (list, tuple)) else None

    if args.architecture == 'cnn_trad_fpool3':
        model = build_cnn_trad_fpool3(num_classes=len(classes), input_shape=input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # optimizer irrelevant for eval
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    elif args.architecture == 'cnn_tpool2':
        model = build_cnn_tpool2(num_classes=len(classes), input_shape=input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # optimizer irrelevant for eval
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    elif args.architecture == 'cnn_tpool2_se':
        model = build_cnn_tpool2_se(num_classes=len(classes), input_shape=input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # optimizer irrelevant for eval
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    elif args.architecture == 'cnn_tpool2_cbam':
        model = build_cnn_tpool2_cbam(num_classes=len(classes), input_shape=input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # optimizer irrelevant for eval
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    else: 
        raise ValueError(f"Unknown architecture: {args.architecture}")
        

    # Load best weights
    if args.use_wandb and args.weights_path_wandb:
        weights_prefix = load_weights(args)
    else:
        weights_prefix = args.weights_path or "checkpoints/best.ckpt"
    model.load_weights(weights_prefix)

    # Evaluate scalar metrics (same loss/acc as training)
    results = model.evaluate(test_ds, verbose=0, return_dict=True)
    # results contains: {"loss": ..., "accuracy": ...}

    # More detailed metrics
    y_true, y_pred, y_prob, x_batches = collect_predictions(model, test_ds)
    top5 = compute_topk_accuracy(y_true, y_prob, k=5)

    if args.use_wandb:
        import wandb
        wandb.log({
            "test/loss": float(results["loss"]),
            "test/accuracy": float(results.get("accuracy", np.mean(y_true == y_pred))),
            "test/top5_accuracy": float(top5),
        })

        log_confusion_and_examples_spectrogram(
                y_true=y_true, y_pred=y_pred, x_batches=x_batches,
                class_names=class_names, num_examples=2
            )
        wandb.finish()

    # Also print for non-wandb usage
    print("Test results:", results)
    print("Test top-5 acc:", top5)


if __name__ == "__main__":
    main()
## CALL it as such: python -m src.evaluate --use_wandb --weights_path_wandb "results/best.ckpt" --download_dir "results/" --wandb_run_id "az4urqv6" # REMEMBER ARCHITECURE AND WANDB RUN ID MUST MATCH THE TRAINING ONES