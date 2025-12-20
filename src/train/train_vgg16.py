"""
Train VGG16 Multitask Model (48x48, CPU-safe)
"""

import os
import tensorflow as tf
from tensorflow import keras

from src.models.multitask_vgg16 import build_multitask_vgg16
from src.data.tf_data import load_train_data, load_val_data
from src.utils.memory import reset_tf

# =====================================================
# Paths
# =====================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "vgg16_48_best.keras")

# =====================================================
# Training
# =====================================================


def main():
    # Clear old graphs / memory
    reset_tf()

    print("📥 Loading training and validation data...")
    train_ds = load_train_data(batch_size=32, augment=True)
    val_ds = load_val_data(batch_size=32)

    print("📐 Building VGG16 multitask model (48x48)...")
    model = build_multitask_vgg16(
        input_shape=(48, 48, 3)
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "age_output": "mae",
            "gender_output": "sparse_categorical_crossentropy",
            "ethnicity_output": "sparse_categorical_crossentropy",
        },
        metrics={
            "gender_output": "accuracy",
            "ethnicity_output": "accuracy",
        }
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )
    ]

    print("🚀 Starting training...\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        callbacks=callbacks
    )

    print(f"\n🎉 Training complete!")
    print(f"📦 Best model saved at:\n{SAVE_PATH}")


if __name__ == "__main__":
    main()
