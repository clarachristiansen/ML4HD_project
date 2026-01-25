import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import wandb

import time
import gc
try:
    import psutil
except ImportError:
    psutil = None


def compute_topk_accuracy(y_true, y_prob, k=5):
    topk = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.any(topk == y_true[:, None], axis=1)
    return float(np.mean(correct))


def collect_predictions(model, ds):
    y_true_list, y_pred_list, y_prob_list = [], [], []
    x_store = []  # optional: to capture some examples (for spectrogram case)

    for b, (x, y) in enumerate(ds):

        probs = model.predict(x, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_np = y.numpy() if isinstance(y, tf.Tensor) else np.asarray(y)
        y_np = y_np.astype(int)

        y_true_list.append(y_np)
        y_pred_list.append(preds)
        y_prob_list.append(probs)

        x_store.append(x)  # keep batch to extract misclassified later (spectrogram)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    return y_true, y_pred, y_prob, x_store

def log_confusion_matrix_image(
    y_true,
    y_pred,
    class_names,
    normalize="true",
    scale_power=-2,              # show values ×10^-2
    figsize=(14, 14),            # BIG squares
    fontsize_cells=7,            # small numbers
    fontsize_labels=12,
    title="Confusion Matrix (Normalized)",
):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Scale values (e.g. 0.37 -> 37 with ×10^-2 note)
    scale = 10 ** (-scale_power)
    cm_scaled = cm * scale

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_scaled,
        display_labels=class_names,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format=".0f",      # integers after scaling
        colorbar=False,
    )

    # Make numbers smaller
    for text in disp.text_.ravel():
        text.set_fontsize(fontsize_cells)

    # Axis labels & ticks
    ax.set_xlabel("Predicted label", fontsize=fontsize_labels)
    ax.set_ylabel("True label", fontsize=fontsize_labels)
    ax.tick_params(axis="both", labelsize=fontsize_labels)

    # Title + scale annotation
    ax.set_title(title, fontsize=fontsize_labels + 2)
    ax.text(
        0.995,
        1.01,
        rf"Values shown in ‰",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize_labels,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
    )

    plt.tight_layout()

    # Log to W&B as real image
    wandb.log({
        "test/confusion_matrix_image": wandb.Image(fig)
    })

    plt.close(fig)



def log_confusion_and_examples_spectrogram(y_true, y_pred, x_batches, class_names, num_examples=2):

    # Confusion matrix plot (interactive)
    log_confusion_matrix_image(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize="true",
        title="Confusion Matrix (Normalized)"
    )

    # Find a couple misclassified examples and log as images
    logged = 0
    idx_global = 0
    for xb in x_batches:
        xb_np = xb.numpy() if isinstance(xb, tf.Tensor) else xb
        batch_size = xb_np.shape[0]

        for i in range(batch_size):
            if y_pred[idx_global] != y_true[idx_global]:
                x = xb_np[i]
                if x.ndim == 3 and x.shape[-1] == 1:
                    x = x[..., 0]

                t = class_names[y_true[idx_global]] if class_names else str(y_true[idx_global])
                p = class_names[y_pred[idx_global]] if class_names else str(y_pred[idx_global])

                fig = plt.figure(figsize=(5, 4))
                plt.imshow(x, aspect="auto", origin="lower")
                plt.title(f"True: {t} | Pred: {p}")
                plt.tight_layout()
                wandb.log({f"test/misclassified_{logged}": wandb.Image(fig)})
                plt.close(fig)

                logged += 1
                if logged >= num_examples:
                    return
            idx_global += 1
        # keep idx_global consistent
        # if we didn't consume all due to early return, that's fine


def log_confusion_and_examples_graph(y_true, y_pred, class_names, misclassified_table_rows):
    import wandb

    # Confusion matrix plot (interactive)
    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=class_names
        )
    })

    # Misclassified examples: log as a Table (graphs are hard to "image" generically)
    columns = ["true_id", "true_name", "pred_id", "pred_name", "notes"]
    table = wandb.Table(columns=columns, data=misclassified_table_rows)
    wandb.log({"test/misclassified_table": table})


def extract_misclassified_graph_rows(ds, y_true, y_pred, class_names, max_rows=2):
    """
    Generic GraphTensor logging: we can't assume how to render it.
    We'll just record true/pred plus lightweight graph stats if accessible.
    """
    rows = []
    idx_global = 0
    for b, (x, y) in enumerate(ds):

        batch_size = int(y.shape[0]) if hasattr(y, "shape") else len(y)
        for i in range(batch_size):
            if y_pred[idx_global] != y_true[idx_global]:
                t_id = int(y_true[idx_global])
                p_id = int(y_pred[idx_global])
                t_name = class_names[t_id] if class_names else str(t_id)
                p_name = class_names[p_id] if class_names else str(p_id)

                notes = ""
                # Try to capture simple GraphTensor stats if present
                try:
                    # GraphTensor usually has sizes, may vary by TF-GNN version
                    sizes = getattr(x, "sizes", None)
                    if sizes is not None:
                        notes = f"sizes={sizes.numpy().tolist()}"
                except Exception:
                    pass

                rows.append([t_id, t_name, p_id, p_name, notes])
                if len(rows) >= max_rows:
                    return rows
            idx_global += 1
    return rows



def load_weights(args):
    """
    Returns checkpoint prefix usable by model.load_weights(prefix).
    Loads from:
      1) W&B Files tab (via run ID)
      2) Local path
    """

    if args.use_wandb and args.wandb_run_id:
        import wandb

        run_path = f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}"

        api = wandb.Api()
        run = api.run(run_path)

        os.makedirs(args.download_dir, exist_ok=True)

        # Find checkpoint files in Files tab
        index_files = [
            f for f in run.files()
            if f.name.endswith(".index") and args.weights_path_wandb in f.name
        ]

        if not index_files:
            raise FileNotFoundError(
                f"No checkpoint .index file matching '{args.weights_path_wandb}' "
                f"found in Files tab of run {run_path}"
            )

        index_file = index_files[0]
        index_file.download(root=args.download_dir, replace=True)

        # Derive prefix and download data shard
        prefix = os.path.join(
            args.download_dir,
            index_file.name.replace(".index", "")
        )

        data_file = index_file.name.replace(".index", ".data-00000-of-00001")
        run.file(data_file).download(root=args.download_dir, replace=True)

        print(f"[OK] Loaded weights from W&B Files, here: {prefix}")
        return prefix

def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)

def get_model_param_stats(model: tf.keras.Model) -> dict:
    total_params = int(model.count_params())
    # float32 params ~4 bytes each; (this is only parameters, not activations)
    param_mem_mb = total_params * 4 / (1024**2)
    return {
        "model/params_total": total_params,
        "model/params_mem_est_mb_fp32": float(param_mem_mb),
    }

def _get_tf_gpu_mem_mb(gpu_device: str = "GPU:0") -> dict:
    """
    Returns current/peak GPU memory if TF exposes it.
    Not guaranteed on all TF builds/drivers.
    """
    out = {}
    try:
        info = tf.config.experimental.get_memory_info(gpu_device)
        # keys usually: 'current', 'peak' in bytes
        out["gpu/mem_current_mb"] = _bytes_to_mb(info.get("current", 0))
        out["gpu/mem_peak_mb"] = _bytes_to_mb(info.get("peak", 0))
    except Exception:
        pass
    return out

def measure_inference_resources(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_batches: int = 50,
    warmup_batches: int = 10,
) -> dict:
    """
    Measures latency + (process) CPU/RAM while running forward passes on `dataset`.
    - Uses psutil if available for CPU/RAM.
    - Uses TF memory_info for GPU if available.
    """
    # Prefer deterministic-ish measurements
    gc.collect()
    tf.keras.backend.clear_session()  # optional; remove if it breaks your workflow

    # Rebuild/ensure model is "built"
    # If model already built & weights loaded, fine.
    # We just need a callable forward pass.

    proc = psutil.Process(os.getpid()) if psutil else None
    if proc:
        proc.cpu_percent(interval=None)  # prime

    # Take a baseline RAM reading (RSS)
    ram_rss_start = proc.memory_info().rss if proc else None
    ram_rss_peak = ram_rss_start if proc else None

    # Warmup (important for TF graph tracing / autotuning)
    it = iter(dataset)
    for _ in range(warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        _ = model(x, training=False)

    # Timed inference
    latencies_ms = []
    cpu_samples = []

    # If you want “per-sample” latency, estimate batch size from first batch.
    batch_sizes = []

    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        # batch size: works for dense tensors; for ragged, you may need tweaks
        try:
            bs = int(tf.shape(x)[0].numpy())
            batch_sizes.append(bs)
        except Exception:
            pass

        t0 = time.perf_counter()
        _ = model(x, training=False)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

        if proc:
            cpu_samples.append(proc.cpu_percent(interval=None))
            rss = proc.memory_info().rss
            ram_rss_peak = rss if (ram_rss_peak is None or rss > ram_rss_peak) else ram_rss_peak

    # Aggregate
    out = {}
    if latencies_ms:
        out["infer/latency_ms_mean_per_batch"] = float(np.mean(latencies_ms))
        out["infer/latency_ms_p50_per_batch"] = float(np.percentile(latencies_ms, 50))
        out["infer/latency_ms_p90_per_batch"] = float(np.percentile(latencies_ms, 90))
        out["infer/latency_ms_p95_per_batch"] = float(np.percentile(latencies_ms, 95))

    if batch_sizes and latencies_ms:
        mean_bs = float(np.mean(batch_sizes))
        out["infer/batch_size_mean"] = mean_bs
        out["infer/latency_ms_mean_per_sample_est"] = float(np.mean(latencies_ms) / max(mean_bs, 1.0))

    if proc and cpu_samples:
        out["cpu/process_cpu_percent_mean"] = float(np.mean(cpu_samples))
        out["cpu/process_cpu_percent_p95"] = float(np.percentile(cpu_samples, 95))

    if proc and ram_rss_start is not None and ram_rss_peak is not None:
        out["ram/rss_start_mb"] = _bytes_to_mb(ram_rss_start)
        out["ram/rss_peak_mb"] = _bytes_to_mb(ram_rss_peak)
        out["ram/rss_increase_mb"] = _bytes_to_mb(ram_rss_peak - ram_rss_start)

    # Optional GPU stats
    out.update(_get_tf_gpu_mem_mb("GPU:0"))

    return out
