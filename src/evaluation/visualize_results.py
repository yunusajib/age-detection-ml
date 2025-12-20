"""
Visualize and compare Base CNN, Improved CNN, and VGG-style models
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from src.evaluation.evaluation import evaluate_model
from src.data.tf_data import load_test_data

# ==================================================
# ✅ PROJECT ROOT (FIXED)
# ==================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)
sys.path.insert(0, PROJECT_ROOT)

# ==================================================
# ✅ MODEL PATHS (FIXED)
# ==================================================
MODELS = {
    "Base_CNN_48x48": os.path.join(PROJECT_ROOT, "models", "best_model.keras"),
    "Improved_CNN_48x48": os.path.join(PROJECT_ROOT, "models", "best_model_improved.keras"),
    "VGG_Style_48x48": os.path.join(PROJECT_ROOT, "models", "vgg16_48_best.keras"),
}

# ==================================================
# Visualization directory
# ==================================================
BASE_VIZ_DIR = os.path.join(PROJECT_ROOT, "visualizations")
os.makedirs(BASE_VIZ_DIR, exist_ok=True)

# ==================================================
# Confusion Matrix Plot
# ==================================================


def plot_confusion_matrix(cm, labels, title, save_path, accuracy):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{int(cm[i, j])}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center"
            )

    ax.set_title(f"{title}\nAccuracy: {accuracy:.2%}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {save_path}")

# ==================================================
# Age Analysis Plot
# ==================================================


def plot_age_analysis(y_true, y_pred, save_path):
    errors = y_pred - y_true

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(errors, bins=50)
    ax[0].set_title("Age Prediction Error Distribution")
    ax[0].set_xlabel("Error (years)")
    ax[0].set_ylabel("Count")

    ax[1].scatter(y_true, y_pred, s=5, alpha=0.4)
    ax[1].plot([0, 100], [0, 100], "--")
    ax[1].set_title("Predicted vs True Age")
    ax[1].set_xlabel("True Age")
    ax[1].set_ylabel("Predicted Age")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {save_path}")

# ==================================================
# MAIN
# ==================================================


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
            print(f"⚠️ Skipping {model_name} (model file not found)")
            continue

        model = keras.models.load_model(model_path, compile=False)
        results = evaluate_model(model, test_ds)

        # Create per-model directory
        model_viz_dir = os.path.join(BASE_VIZ_DIR, model_name)
        os.makedirs(model_viz_dir, exist_ok=True)

        # Gender
        plot_confusion_matrix(
            results["gender"]["confusion_matrix"],
            labels=["Male", "Female"],
            title=f"{model_name} – Gender",
            save_path=os.path.join(model_viz_dir, "gender_cm.png"),
            accuracy=results["gender"]["accuracy"],
        )

        # Ethnicity
        plot_confusion_matrix(
            results["ethnicity"]["confusion_matrix"],
            labels=[f"Class {i}" for i in range(
                results["ethnicity"]["confusion_matrix"].shape[0]
            )],
            title=f"{model_name} – Ethnicity",
            save_path=os.path.join(model_viz_dir, "ethnicity_cm.png"),
            accuracy=results["ethnicity"]["accuracy"],
        )

        # Age
        plot_age_analysis(
            results["y_true"]["age"],
            results["y_pred"]["age"],
            save_path=os.path.join(model_viz_dir, "age_analysis.png"),
        )

    print("\n🎉 ALL MODELS VISUALIZED")
    print(f"📁 Output directory: {BASE_VIZ_DIR}")


if __name__ == "__main__":
    main()
