import argparse
import os
import tensorflow as tf
import wandb
from .preprocessing import get_datasets
from .model_sainath import build_cnn_trad_fpool3, build_cnn_tpool2
from .model_se import build_cnn_tpool2_se
from .model_inception import build_cnn_inception_1, build_cnn_inception_2
from .model_autoencoder import build_cnn_autoencoder_for_svm, extract_features, build_svm
from .utils_train import StandardTrainer, InceptionTrainer, EncoderSVMTrainer
from .wandb import WandbMetricsCallback, WandbMetricsCallbackInception, WandbMetricsCallbackAE

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", type=str, default="cnn_trad_fpool3")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_entity", type=str, default="clara-christiansen-danmarks-tekniske-universitet-dtu")
    p.add_argument("--wandb_project", type=str, default="ML4HD_project")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--frames", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False, frames=args.frames)

    if args.architecture == "cnn_trad_fpool3":
        trainer = StandardTrainer(build_cnn_trad_fpool3)
    elif args.architecture == "cnn_tpool2":
        trainer = StandardTrainer(build_cnn_tpool2)
    elif args.architecture == "cnn_tpool2_se":
        trainer = StandardTrainer(build_cnn_tpool2_se)
    elif args.architecture == "cnn_inception_1":
        trainer = InceptionTrainer(build_cnn_inception_1, num_heads=1)
    elif args.architecture == "cnn_inception_2":
        trainer = InceptionTrainer(build_cnn_inception_2, num_heads=2)
    elif args.architecture == "encoder_svm":
        trainer = EncoderSVMTrainer(build_cnn_autoencoder_for_svm, extract_features, build_svm)
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")

    best_prefix = "results/best.ckpt"  

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

    if args.use_wandb:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            settings=wandb.Settings(start_method="thread"),
            config=args.__dict__
        )
        if trainer.__class__.__name__ == "InceptionTrainer":
            callbacks += [
                WandbMetricsCallbackInception(num_heads=trainer.num_heads),
            ]
        elif trainer.__class__.__name__ == "EncoderSVMTrainer":
            callbacks = [
                WandbMetricsCallbackAE(),
            ]
        else:
            callbacks += [
                WandbMetricsCallback(),
            ]

    result = trainer.train(args, train_ds, val_ds, classes, callbacks)

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
    print("Training complete.")

    
if __name__ == "__main__":
    main()
