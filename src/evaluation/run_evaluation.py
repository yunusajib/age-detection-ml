from src.evaluation.evaluation import evaluate_model
from src.data.tf_data import load_test_data
import os
import sys
import tensorflow as tf
from tensorflow import keras

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# Define the custom loss class

class WeightedSparseCategoricalCrossentropy(keras.losses.Loss):
    """Custom loss that applies class weights."""

    def __init__(self, class_weights=None, name="weighted_sparse_categorical_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = class_weights if class_weights is not None else {}

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, "int32")
        y_true = tf.reshape(y_true, (-1,))

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss = loss_fn(y_true, y_pred)

        weights = tf.gather(
            list(self.class_weights.values()),
            y_true
        )

        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"class_weights": self.class_weights})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


MODEL_PATH = os.path.join(PROJECT_ROOT, "models/best_model_improved.keras")


def main():
    print(f"🔍 Loading model from: {MODEL_PATH}")
    print(f"📁 File exists: {os.path.exists(MODEL_PATH)}")

    print("📥 Loading test data...")
    test_ds = load_test_data(batch_size=32)

    print("🧠 Loading model with custom objects...")
    # Load model with custom_objects parameter
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'WeightedSparseCategoricalCrossentropy': WeightedSparseCategoricalCrossentropy}
    )

    print("🔍 Evaluating model...\n")
    results = evaluate_model(model, test_ds)

    # Print results
    print("=" * 50)
    print("AGE PREDICTION METRICS")
    print("=" * 50)
    for k, v in results["age"].items():
        print(f"{k}: {v:.4f}")

    print("\n" + "=" * 50)
    print("GENDER CLASSIFICATION METRICS")
    print("=" * 50)
    for k, v in results["gender"].items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    print(f"\nConfusion Matrix:\n{results['gender']['confusion_matrix']}")

    print("\n" + "=" * 50)
    print("ETHNICITY CLASSIFICATION METRICS")
    print("=" * 50)
    for k, v in results["ethnicity"].items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    print(f"\nConfusion Matrix:\n{results['ethnicity']['confusion_matrix']}")


if __name__ == "__main__":
    main()
