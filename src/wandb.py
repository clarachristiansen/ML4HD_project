import wandb
import tensorflow as tf
import os

class WandbMetricsCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        wandb.log(
            {
                "train/loss": logs["loss"],
                "train/accuracy": logs.get("accuracy"),
            },
            commit=False,
        )

    def on_epoch_end(self, epoch, logs=None):
        wandb.log(
            {
                "epoch": epoch,
                "val/loss": logs.get("val_loss"),
                "val/accuracy": logs.get("val_accuracy"),
            }
        )


class WandbMetricsCallbackInception(tf.keras.callbacks.Callback):
    def __init__(self, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads

    def on_train_begin(self, logs=None):
        # Ensure epoch is the step axis for these metrics
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        log_dict = {"epoch": epoch}

        # inception_1 
        if logs.get("loss") is not None:
            log_dict["train/loss"] = float(logs["loss"])
        if logs.get("accuracy") is not None:
            log_dict["train/accuracy"] = float(logs["accuracy"])
        if logs.get("val_loss") is not None:
            log_dict["val/loss"] = float(logs["val_loss"])
        if logs.get("val_accuracy") is not None:
            log_dict["val/accuracy"] = float(logs["val_accuracy"])

        # inception_x where x > 1
        for i in range(1, self.num_heads + 1):
            tr_l = logs.get(f"softmax_{i}_loss")
            tr_a = logs.get(f"softmax_{i}_accuracy")
            va_l = logs.get(f"val_softmax_{i}_loss")
            va_a = logs.get(f"val_softmax_{i}_accuracy")

            if tr_l is not None:
                log_dict[f"train/softmax_{i}_loss"] = float(tr_l)
            if tr_a is not None:
                log_dict[f"train/softmax_{i}_accuracy"] = float(tr_a)
            if va_l is not None:
                log_dict[f"val/softmax_{i}_loss"] = float(va_l)
            if va_a is not None:
                log_dict[f"val/softmax_{i}_accuracy"] = float(va_a)

        wandb.log(log_dict)


class WandbMetricsCallbackAE(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        wandb.log(
            {
                "train/loss": logs["loss"],
            },
            commit=False,
        )
    
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(
            {
                "epoch": epoch,
                "val/loss": logs.get("val_loss"),
            }
        )

class WandbCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        every_n_epochs: int = 5,
        artifact_name: str = "model-weights",
        artifact_type: str = "model",
        save_format: str = "ckpt",  # or "h5"
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        self.save_format = save_format

    def on_epoch_end(self, epoch, logs=None):
        # Epoch numbers are 0-based internally
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        epoch_num = epoch + 1
        run = wandb.run
        if run is None:
            return

        save_dir = f"checkpoints/epoch_{epoch_num}"
        os.makedirs(save_dir, exist_ok=True)

        if self.save_format == "ckpt":
            path = os.path.join(save_dir, "ckpt")
            self.model.save_weights(path)
        elif self.save_format == "h5":
            path = os.path.join(save_dir, "weights.h5")
            self.model.save_weights(path)
        else:
            raise ValueError("save_format must be 'ckpt' or 'h5'")

        artifact = wandb.Artifact(
            name=f"{self.artifact_name}-epoch-{epoch_num}",
            type=self.artifact_type,
            metadata={
                "epoch": epoch_num,
                "val_accuracy": logs.get("val_accuracy") if logs else None,
                "val_loss": logs.get("val_loss") if logs else None,
            },
        )

        artifact.add_dir(save_dir)
        run.log_artifact(artifact)

