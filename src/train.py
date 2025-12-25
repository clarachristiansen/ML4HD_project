import argparse
import os
import tensorflow as tf
import wandb
from .preprocessing import get_datasets
from .model_sainath import build_cnn_trad_fpool3, build_cnn_tpool2
from .wandb import WandbMetricsCallback  # your custom callback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", type=str, default="cnn_trad_fpool3")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="ML4HDproject")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False)

    if args.architecture == "cnn_trad_fpool3":
        model = build_cnn_trad_fpool3(num_classes=len(classes))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    elif args.architecture == "cnn_tpool2":
        model = build_cnn_tpool2(num_classes=len(classes))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    # MAYBE MORE!!!
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    best_prefix = "results/best.ckpt"  # produces best.ckpt.index + best.ckpt.data-...

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_prefix,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ---- Optional W&B ----
    if args.use_wandb:
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=args.wandb_project,
            name=args.wandb_run_name,
            settings=wandb.Settings(start_method="thread"),
            config={
                "epochs": args.epochs,
                "lr": args.lr,
                "num_classes": len(classes),
                "job_id": os.environ.get("LSB_JOBID"),
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
        idx = best_prefix + ".index"
        dat = best_prefix + ".data-00000-of-00001"
        try:
            wandb.save(idx)
            wandb.save(dat)
            print('INFO: Best weights files saved locally and to wandb (files tab).')
        except Exception as e:
            print("WARN: saving best weights files to wandb failed. Error:", repr(e))
        wandb.finish()

if __name__ == "__main__":
    main()
