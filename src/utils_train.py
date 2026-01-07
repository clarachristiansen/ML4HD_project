from typing import Any, Callable, Dict, Optional
import tensorflow as tf

class Trainer():
    """Minimal interface so train.py can treat all architectures uniformly."""
    def train(self, args, train_ds, val_ds, test_ds, classes, callbacks, wandb_run=None) -> dict:
        raise NotImplementedError

class StandardTrainer(Trainer):
    """Standard Keras model trainer."""
    def __init__(self, build_function: Callable):
        self.build_function = build_function

    def train(self, args, train_ds, val_ds, classes, callbacks) -> dict:
        model = self.build_function(input_shape=(args.frames, 40, 1), num_classes=len(classes))
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
