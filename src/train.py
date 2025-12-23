import argparse
import tensorflow as tf
from .preprocessing import get_datasets
from .model_sainath import build_cnn_trad_fpool3  
from .wandb import WandbMetricsCallback  # your custom callback

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", type=str, default="cnn_trad_fpool3")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="ML4HD_project")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False)

    if args.architecture == "cnn_trad_fpool3":
        model = build_cnn_trad_fpool3(num_classes=len(classes))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    # MAYBE MORE!!!
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    # ---- Optional W&B ----
    if args.use_wandb:
        import wandb
        from .wandb import WandbMetricsCallback, WandbCheckpointCallback  # your custom callbacks

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "lr": args.lr,
                "num_classes": len(classes),
            },
        )

        # You can use either WandbCallback OR your WandbMetricsCallback.
        # If you keep your custom one, set WandbCallback to minimal or omit it.
        callbacks += [
            # Optional: logs keras metrics automatically; nice but can overlap with custom logs
            # WandbCallback(save_model=False),
            WandbMetricsCallback(),  # your explicit logging callback
        ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ---- Optional W&B finalize ----
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
