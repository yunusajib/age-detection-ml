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
# ✅ PROJECT ROOT
# ==================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)
sys.path.insert(0, PROJECT_ROOT)

# ==================================================
# ✅ MODEL PATHS
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
# 🔥 NEW: MODEL COMPARISON PLOTS
# ==================================================


def plot_model_comparison(all_results):
    models = list(all_results.keys())

    gender_acc = [all_results[m]["gender"]["accuracy"] for m in models]
    ethnicity_acc = [all_results[m]["ethnicity"]["accuracy"] for m in models]
    age_mae = [all_results[m]["age"]["MAE"] for m in models]

    # ---- Accuracy Comparison ----
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, gender_acc, width, label="Gender Accuracy")
    plt.bar(x + width/2, ethnicity_acc, width, label="Ethnicity Accuracy")

    plt.xticks(x, models, rotation=15)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison: Classification Accuracy")
    plt.legend()

    acc_path = os.path.join(BASE_VIZ_DIR, "model_accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {acc_path}")

    # ---- Age MAE Comparison ----
    plt.figure(figsize=(8, 4))
    plt.bar(models, age_mae)
    plt.ylabel("MAE (years)")
    plt.title("Model Comparison: Age Prediction Error")
    plt.xticks(rotation=15)

    mae_path = os.path.join(BASE_VIZ_DIR, "model_age_mae_comparison.png")
    plt.tight_layout()
    plt.savefig(mae_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {mae_path}")

# ==================================================
# MAIN
# ==================================================


def main():
    print("\n📥 Loading test dataset...")
    test_ds = load_test_data(batch_size=32)

    all_results = {}

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
        all_results[model_name] = results

        # Per-model directory
        model_viz_dir = os.path.join(BASE_VIZ_DIR, model_name)
        os.makedirs(model_viz_dir, exist_ok=True)

        # Gender CM
        plot_confusion_matrix(
            results["gender"]["confusion_matrix"],
            labels=["Male", "Female"],
            title=f"{model_name} – Gender",
            save_path=os.path.join(model_viz_dir, "gender_cm.png"),
            accuracy=results["gender"]["accuracy"],
        )

        # Ethnicity CM
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

    # 🔥 NEW: comparison plots
    plot_model_comparison(all_results)

    print("\n🎉 ALL MODELS VISUALIZED & COMPARED")
    print(f"📁 Output directory: {BASE_VIZ_DIR}")


if __name__ == "__main__":
    main()
