from src.data.tf_data import load_train_data, load_val_data
from src.models.multitask_cnn_improved import build_multitask_model
from tensorflow import keras
import sys
import os
import numpy as np

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================
# Configuration
# ============================

EPOCHS = 40
BATCH_SIZE = 32
DROPOUT_RATE = 0.5  # Adjustable dropout rate

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# Class Weights (RUN calculate_class_weights.py first!)
# ============================
# These weights help the model pay more attention to minority classes
# Replace these with YOUR calculated weights from calculate_class_weights.py

# Replace these lines in train_improved.py:

ethnicity_class_weights = {
    0: 0.47045398164227237,
    1: 1.0474454570560618,
    2: 1.3802037845705968,
    3: 1.1923294561458662,
    4: 2.805325443786982
}

gender_class_weights = {
    0: 0.9568113017154389,
    1: 1.0472719240114867
}
# ============================
# Custom Loss with Class Weights
# ============================


class WeightedSparseCategoricalCrossentropy(keras.losses.Loss):
    """Custom loss that applies class weights."""

    def __init__(self, class_weights, name="weighted_sparse_categorical_crossentropy"):
        super().__init__(name=name)
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        # Get class weights for each sample
        y_true = keras.ops.cast(y_true, "int32")
        y_true = keras.ops.reshape(y_true, (-1,))

        # Calculate loss
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss = loss_fn(y_true, y_pred)

        # Apply weights
        weights = keras.ops.take(
            list(self.class_weights.values()),
            y_true
        )

        weighted_loss = loss * weights
        return keras.ops.mean(weighted_loss)

# ============================
# Main Training Script
# ============================


def main():

    print("📥 Loading datasets...")
    train_ds = load_train_data(batch_size=32, augment=True)
    val_ds = load_val_data(batch_size=BATCH_SIZE)

    print(f"🧠 Building model with dropout rate: {DROPOUT_RATE}...")
    model = build_multitask_model(dropout_rate=DROPOUT_RATE)

    # Create weighted losses
    ethnicity_loss = WeightedSparseCategoricalCrossentropy(
        class_weights=ethnicity_class_weights,
        name="ethnicity_weighted_loss"
    )

    gender_loss = WeightedSparseCategoricalCrossentropy(
        class_weights=gender_class_weights,
        name="gender_weighted_loss"
    )

    # Compile model with weighted losses
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-4),  # Lower learning rate
        loss={
            "age_output": keras.losses.MeanSquaredError(),
            "gender_output": gender_loss,
            "ethnicity_output": ethnicity_loss,
        },
        metrics={
            "age_output": ["mae"],
            "gender_output": ["accuracy"],
            "ethnicity_output": ["accuracy"],
        }
    )

    model.summary()

    # ============================
    # Callbacks
    # ============================

    checkpoint_path = os.path.join(MODEL_DIR, "best_model_improved.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs_improved")
        )
    ]

    # ============================
    # Training
    # ============================

    print("\n🚀 Starting training with improvements...\n")
    print("✅ Dropout enabled")
    print("✅ Class weights applied")
    print("✅ Lower learning rate")
    print("✅ Learning rate reduction on plateau\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ============================
    # Save final model
    # ============================

    final_model_path = os.path.join(MODEL_DIR, "final_model_improved.keras")
    model.save(final_model_path)

    print(f"\n🎉 Training complete!")
    print(f"📦 Best model saved at: {checkpoint_path}")
    print(f"📦 Final model saved at: {final_model_path}")

    # Print final metrics
    print("\n" + "="*50)
    print("FINAL TRAINING METRICS")
    print("="*50)
    final_metrics = history.history
    print(f"Final train loss: {final_metrics['loss'][-1]:.4f}")
    print(f"Final val loss: {final_metrics['val_loss'][-1]:.4f}")
    print(f"Best val loss: {min(final_metrics['val_loss']):.4f}")


if __name__ == "__main__":
    main()
