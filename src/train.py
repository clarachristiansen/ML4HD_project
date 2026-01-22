import argparse
import os
import random
import numpy as np
import tensorflow as tf
import wandb
from .preprocessing import get_datasets
from .model_sainath import build_cnn_trad_fpool3, build_cnn_tpool2
from .model_se import build_cnn_tpool2_se
from .model_cbam import build_cnn_tpool2_cbam
from .model_inception import build_cnn_inception_1, build_cnn_inception_2
from .model_inception_attention import build_cnn_inception_1_attention, build_cnn_inception_2_attention
from .utils_train import StandardTrainer, InceptionTrainer
from .wandb import WandbMetricsCallback, WandbMetricsCallbackInception

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
    p.add_argument("--final_data", type=str, default='logmel_spectrogram', help="Type of input features: 'logmel_spectrogram', 'logmel_spectrogram_bins', 'mel_pcen_a', 'mel_pcen_b' or 'mfccs'")
    p.add_argument("--seed", type=int, default=24, help="Random seed for reproducibility")
    return p.parse_args()

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.final_data == 'mfccs':
        channels = 120
    else:
        channels = 40

    train_ds, val_ds, test_ds, classes, background_noise_files = get_datasets(repeat_train=False, frames=args.frames, final_data=args.final_data)
    test = train_ds.take(1)  # Force dataset initialization
    print("Training instance shape:", tf.shape(next(iter(test))[0]))

    if args.architecture == "cnn_trad_fpool3":
        trainer = StandardTrainer(build_cnn_trad_fpool3, channels=channels)
    elif args.architecture == "cnn_tpool2":
        trainer = StandardTrainer(build_cnn_tpool2, channels=channels)
    elif args.architecture == "cnn_tpool2_se":
        trainer = StandardTrainer(build_cnn_tpool2_se, channels=channels)
    elif args.architecture == "cnn_tpool2_cbam":
        trainer = StandardTrainer(build_cnn_tpool2_cbam, channels=channels)
    elif args.architecture == "cnn_inception_1":
        trainer = InceptionTrainer(build_cnn_inception_1, channels=channels, num_heads=1)
    elif args.architecture == "cnn_inception_2":
        trainer = InceptionTrainer(build_cnn_inception_2, channels=channels, num_heads=2)
    elif args.architecture == "cnn_inception_1_attention":
        trainer = InceptionTrainer(build_cnn_inception_1_attention, channels=channels, num_heads=1)
    elif args.architecture == "cnn_inception_2_attention":
        trainer = InceptionTrainer(build_cnn_inception_2_attention, channels=channels, num_heads=2)
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
