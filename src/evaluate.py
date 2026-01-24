# evaluate.py
import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import wandb
from .utils_evaluate import compute_topk_accuracy, collect_predictions, load_weights,log_confusion_and_examples_spectrogram, log_confusion_and_examples_graph, extract_misclassified_graph_rows, get_model_param_stats, measure_inference_resources
from .preprocessing import get_datasets
from .model_sainath import build_cnn_trad_fpool3, build_cnn_tpool2
from .model_se import build_cnn_tpool2_se
from .model_cbam import build_cnn_tpool2_cbam
from .model_inception import build_cnn_inception_1, build_cnn_inception_2
from .model_inception_attention import build_cnn_inception_1_attention, build_cnn_inception_2_attention


import time
import gc
import os
import numpy as np
import tensorflow as tf

try:
    import psutil
except ImportError:
    psutil = None


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
    print(len(classes), "classes:", classes)
    if 'inception_2' in args.architecture:
        if args.architecture == 'cnn_inception_2':
            model = build_cnn_inception_2(num_classes=len(classes), input_shape=input_shape, num_heads=2)
        elif args.architecture == 'cnn_inception_2_attention':
            model = build_cnn_inception_2_attention(num_classes=len(classes), input_shape=input_shape, num_heads=2)
        model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss={
                    f'softmax_{i}': tf.keras.losses.SparseCategoricalCrossentropy()
                    for i in range(1, 2 + 1)
                },
                loss_weights={
                    f"softmax_{i}": 0.3 if i < 2 else 1.0
                    for i in range(1, 2 + 1)
                },
                metrics={
                    f"softmax_{i}": ["accuracy"]
                    for i in range(1, 2 + 1)
                }
        )
        tf.keras.utils.plot_model(model, show_shapes=True, to_file="results/model.png")
        test_ds = test_ds.map(lambda x, y: (x, {f"softmax_{i}": y for i in range(1, 2 + 1)}))

        
    else:
        if args.architecture == 'cnn_trad_fpool3':
            model = build_cnn_trad_fpool3(num_classes=len(classes), input_shape=input_shape)
        elif args.architecture == 'cnn_tpool2':
            model = build_cnn_tpool2(num_classes=len(classes), input_shape=input_shape)
        elif args.architecture == 'cnn_tpool2_se':
            model = build_cnn_tpool2_se(num_classes=len(classes), input_shape=input_shape)
        elif args.architecture == 'cnn_tpool2_cbam':
            model = build_cnn_tpool2_cbam(num_classes=len(classes), input_shape=input_shape)
        elif args.architecture == 'cnn_inception_1':
            model = build_cnn_inception_1(num_classes=len(classes), input_shape=input_shape, num_heads=1)
        elif args.architecture == 'cnn_inception_1_attention':
            model = build_cnn_inception_1_attention(num_classes=len(classes), input_shape=input_shape, num_heads=1)
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")
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


    ######


        # ---- Model size stats
    size_stats = get_model_param_stats(model)

    # ---- Inference resource stats (run on test_ds)
    # You can also use a version of test_ds with batch_size=1 for true latency.
    resource_stats = measure_inference_resources(
        model=model,
        dataset=test_ds,
        num_batches=50,
        warmup_batches=10,
    )

    if args.use_wandb:
        wandb.log({**size_stats, **resource_stats})
    else:
        print("Model size stats:", size_stats)
        print("Inference resource stats:", resource_stats)

    ##############



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

# python -m src.evaluate --architecture "cnn_tpool2" --frames 98 --final_data "logmel_spectrogram" --download_dir "results/" --weights_path_wandb "results/checkpoint_9w8m9jle.ckpt" --wandb_run_id "9w8m9jle" --use_wandb