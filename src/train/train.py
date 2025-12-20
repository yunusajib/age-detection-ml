from src.data.tf_data import load_train_data, load_val_data
from src.models.multitask_cnn import build_multitask_model
from tensorflow import keras
import sys
import os

# Add project root to Python path FIRST (before importing from src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Now import from src (AFTER sys.path is set)


# ============================
# Configuration
# ============================

EPOCHS = 40
BATCH_SIZE = 32

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================
# Main Training Script
# ============================

def main():

    print("📥 Loading datasets...")
    train_ds = load_train_data(batch_size=BATCH_SIZE)
    val_ds = load_val_data(batch_size=BATCH_SIZE)

    print("🧠 Building model...")
    model = build_multitask_model()

    # Compile model with correct output names
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "age_output": keras.losses.MeanSquaredError(),
            "gender_output": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "ethnicity_output": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")

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
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs")
        )
    ]

    # ============================
    # Training
    # ============================

    print("\n🚀 Starting training...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ============================
    # Save final model
    # ============================

    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
    model.save(final_model_path)

    print(f"\n🎉 Training complete!")
    print(f"📦 Best model saved at: {checkpoint_path}")
    print(f"📦 Final model saved at: {final_model_path}")


if __name__ == "__main__":
    main()
