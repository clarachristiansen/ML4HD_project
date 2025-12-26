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
from .model_sainath import build_cnn_trad_fpool3

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ML4HDproject")
    p.add_argument("--wandb_entity", type=str, default="clara-christiansen-danmarks-tekniske-universitet-dtu")
    p.add_argument("--wandb_run_id", type=str, default=None, help="If set, resume/log into this exact run id")
    # Where to load best weights from
    p.add_argument("--download_dir", type=str, default=None, help="Local checkpoint prefix, e.g. checkpoints/best.ckpt")
    p.add_argument("--weights_path_wandb", type=str, default=None, help="W&B file name, e.g. model_best_weights_sainath")

    # Data / model configuration
    p.add_argument("--input_type", choices=["spectrogram", "graph"], default="spectrogram")
    return p.parse_args()


def main():
    args = parse_args()
    print("Evaluation args:", args)
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

    train_ds, val_ds, test_ds, classes, _ = get_datasets(repeat_train=False)
    class_names = list(classes) if isinstance(classes, (list, tuple)) else None

    model = build_cnn_trad_fpool3(num_classes=len(classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # optimizer irrelevant for eval
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

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

        if args.input_type == "spectrogram":
            log_confusion_and_examples_spectrogram(
                y_true=y_true, y_pred=y_pred, x_batches=x_batches,
                class_names=class_names, num_examples=2
            )
        else:
            # GraphTensor: log a table of misclassified rows + confusion matrix
            mis_rows = extract_misclassified_graph_rows(
                test_ds, y_true, y_pred, class_names, max_rows=2
            )
            log_confusion_and_examples_graph(y_true, y_pred, class_names, mis_rows)

        wandb.finish()

    # Also print for non-wandb usage
    print("Test results:", results)
    print("Test top-5 acc:", top5)


if __name__ == "__main__":
    main()
