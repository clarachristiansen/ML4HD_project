from typing import Any, Callable, Dict, Optional
import wandb
import tensorflow as tf

class Trainer():
    """Minimal interface so train.py can treat all architectures uniformly."""
    def train(self, args, train_ds, val_ds, test_ds, classes, callbacks, wandb_run=None) -> dict:
        raise NotImplementedError

class StandardTrainer(Trainer):
    """Standard Keras model trainer."""
    def __init__(self, build_function: Callable, channels: int = 40):
        self.build_function = build_function
        self.channels = channels

    def train(self, args, train_ds, val_ds, classes, callbacks) -> dict:
        model = self.build_function(input_shape=(args.frames, self.channels, 1), num_classes=len(classes))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
        )
        return {'model': model, 'history': history}
    
class InceptionTrainer(Trainer):
    """Trainer for inception architectures with different number of heads."""
    def __init__(self, build_function: Callable, num_heads: int = 1, channels: int = 40):
        self.build_function = build_function
        self.num_heads = num_heads
        self.channels = channels

    def train(self, args, train_ds, val_ds, classes, callbacks) -> dict:
        model = self.build_function(input_shape=(args.frames, self.channels, 1), num_classes=len(classes))
    
        if self.num_heads == 1:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            train_ds_heads = train_ds
            val_ds_heads = val_ds
        else: 
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss={
                    f'softmax_{i}': tf.keras.losses.SparseCategoricalCrossentropy()
                    for i in range(1, self.num_heads + 1)
                },
                loss_weights={
                    f"softmax_{i}": 0.3 if i < self.num_heads else 1.0
                    for i in range(1, self.num_heads + 1)
                },
                metrics={
                    f"softmax_{i}": ["accuracy"]
                    for i in range(1, self.num_heads + 1)
                }
            )

            train_ds_heads = train_ds.map(lambda x, y: (x, {f"softmax_{i}": y for i in range(1, self.num_heads + 1)}))
            val_ds_heads = val_ds.map(lambda x, y: (x, {f"softmax_{i}": y for i in range(1, self.num_heads + 1)}))

        history = model.fit(
            train_ds_heads,
            validation_data=val_ds_heads,
            epochs=args.epochs,
            callbacks=callbacks,
        )
        return {'model': model, 'history': history}