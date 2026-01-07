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
    
    def on_epoch_end(self, epoch, logs=None):
        log_dict = {
            "epoch": epoch,
            "val/loss": logs.get("val_loss"),
        }
        for i in range(1, self.num_heads + 1):
            head_loss = logs.get(f'val_softmax_{i}_loss')
            head_accuracy = logs.get(f'val_softmax_{i}_accuracy')
            if head_loss is not None:
                log_dict[f"val/softmax_{i}_loss"] = head_loss
            if head_accuracy is not None:
                log_dict[f"val/softmax_{i}_accuracy"] = head_accuracy

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

