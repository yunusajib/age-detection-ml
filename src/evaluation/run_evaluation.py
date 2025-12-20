import os
import sys
from tensorflow import keras
from src.evaluation.evaluation import evaluate_model
from src.data.tf_data import load_test_data

# --------------------------------------------------
# Project root
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Models to evaluate
# --------------------------------------------------
MODELS = {
    "Improved CNN (48x48)": os.path.join(
        PROJECT_ROOT, "models/best_model_improved.keras"
    ),
    "VGG-style (48x48)": os.path.join(
        PROJECT_ROOT, "models/vgg16_48_best.keras"
    ),
}


def print_results(title, results):
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)

    # ---------------- AGE ----------------
    print("\nAGE PREDICTION")
    for k, v in results["age"].items():
        print(f"{k}: {v:.4f}")

    # ---------------- GENDER ----------------
    print("\nGENDER CLASSIFICATION")
    for k, v in results["gender"].items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    print("Confusion Matrix:\n", results["gender"]["confusion_matrix"])

    # ---------------- ETHNICITY ----------------
    print("\nETHNICITY CLASSIFICATION")
    for k, v in results["ethnicity"].items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    print("Confusion Matrix:\n", results["ethnicity"]["confusion_matrix"])


def main():
    print("\n📥 Loading test dataset...")
    test_ds = load_test_data(batch_size=32)

    for model_name, model_path in MODELS.items():
        print("\n" + "#" * 70)
        print(f"🔍 Evaluating: {model_name}")
        print(f"📁 Path: {model_path}")
        print(f"📁 Exists: {os.path.exists(model_path)}")
        print("#" * 70)

        if not os.path.exists(model_path):
            print("❌ Model not found, skipping.")
            continue

        model = keras.models.load_model(model_path, compile=False)

        results = evaluate_model(model, test_ds)

        print_results(model_name, results)


if __name__ == "__main__":
    main()
