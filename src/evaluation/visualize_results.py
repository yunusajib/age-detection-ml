"""
Visualization script for model evaluation results.
Run this after evaluating your model to generate plots.
"""

from src.evaluation.evaluation import evaluate_model
from src.data.tf_data import load_test_data
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# Create visualizations directory
VIZ_DIR = os.path.join(PROJECT_ROOT, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)


# Custom loss class for loading model
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


def plot_confusion_matrix(cm, labels, title, filename, accuracy):
    """
    Plot and save confusion matrix as heatmap.

    Args:
        cm: Confusion matrix array
        labels: List of class labels
        title: Plot title
        filename: Output filename
        accuracy: Overall accuracy to display
    """
    plt.figure(figsize=(10, 8))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})

    plt.title(f'{title}\nOverall Accuracy: {accuracy:.2%}',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Save
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def plot_age_distribution(y_true, y_pred, filename):
    """
    Plot age prediction error distribution.

    Args:
        y_true: True ages
        y_pred: Predicted ages
        filename: Output filename
    """
    errors = y_pred - y_true

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Error distribution histogram
    axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='red', linestyle='--',
                       linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Prediction Error (years)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Age Prediction Error Distribution',
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Scatter plot: Predicted vs True
    axes[0, 1].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0, 1].plot([0, 100], [0, 100], 'r--', linewidth=2,
                    label='Perfect Prediction')
    axes[0, 1].set_xlabel('True Age', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Age', fontsize=11)
    axes[0, 1].set_title('Predicted vs True Age',
                         fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Error by age group
    age_bins = [0, 18, 30, 50, 70, 120]
    age_labels = ['0-18', '19-30', '31-50', '51-70', '70+']
    y_true_binned = np.digitize(y_true, age_bins)

    mae_by_age = []
    for i in range(1, len(age_bins)):
        mask = y_true_binned == i
        if mask.sum() > 0:
            mae_by_age.append(np.mean(np.abs(errors[mask])))
        else:
            mae_by_age.append(0)

    axes[1, 0].bar(age_labels, mae_by_age, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Age Group', fontsize=11)
    axes[1, 0].set_ylabel('Mean Absolute Error (years)', fontsize=11)
    axes[1, 0].set_title('MAE by Age Group', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # 4. Summary statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    median_error = np.median(np.abs(errors))

    stats_text = f"""
    Summary Statistics:
    
    MAE:  {mae:.2f} years
    RMSE: {rmse:.2f} years
    Median Error: {median_error:.2f} years
    
    Within ±5 years: {(np.abs(errors) <= 5).sum() / len(errors) * 100:.1f}%
    Within ±10 years: {(np.abs(errors) <= 10).sum() / len(errors) * 100:.1f}%
    Within ±15 years: {(np.abs(errors) <= 15).sum() / len(errors) * 100:.1f}%
    """

    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.suptitle('Age Prediction Analysis', fontsize=16,
                 fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def plot_comparison_metrics(filename):
    """
    Plot baseline vs improved model comparison.

    Args:
        filename: Output filename
    """
    metrics = {
        'Age MAE\n(years)': [7.32, 7.65],
        'Gender\nAccuracy': [0.613, 0.712],
        'Ethnicity\nAccuracy': [0.429, 0.547],
        'Ethnicity\nF1-Score': [0.175, 0.387]
    }

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    baseline_values = [metrics[k][0] for k in metrics]
    improved_values = [metrics[k][1] for k in metrics]

    bars1 = ax.bar(x - width/2, baseline_values, width,
                   label='Baseline', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, improved_values, width,
                   label='Improved', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline vs Improved Model Performance',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def main():
    """Main visualization pipeline."""

    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Load model and test data
    print("\n📥 Loading test data...")
    test_ds = load_test_data(batch_size=32)

    print("🧠 Loading improved model...")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models/best_model_improved.keras")

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'WeightedSparseCategoricalCrossentropy': WeightedSparseCategoricalCrossentropy}
    )

    print("🔍 Evaluating model...")
    results = evaluate_model(model, test_ds)

    # Extract results
    y_age_true, y_gender_true, y_eth_true = results['y_true']
    y_age_pred, y_gender_pred, y_eth_pred = results['y_pred']

    print("\n📊 Generating visualizations...\n")

    # 1. Gender confusion matrix
    plot_confusion_matrix(
        cm=results['gender']['confusion_matrix'],
        labels=['Male', 'Female'],
        title='Gender Classification',
        filename='gender_confusion_matrix.png',
        accuracy=results['gender']['accuracy']
    )

    # 2. Ethnicity confusion matrix
    ethnicity_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    plot_confusion_matrix(
        cm=results['ethnicity']['confusion_matrix'],
        labels=ethnicity_labels,
        title='Ethnicity Classification (5 Classes)',
        filename='ethnicity_confusion_matrix.png',
        accuracy=results['ethnicity']['accuracy']
    )

    # 3. Age prediction analysis
    plot_age_distribution(
        y_true=y_age_true,
        y_pred=y_age_pred,
        filename='age_prediction_analysis.png'
    )

    # 4. Baseline vs Improved comparison
    plot_comparison_metrics(filename='baseline_vs_improved.png')

    print("\n" + "=" * 60)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"📁 Saved to: {VIZ_DIR}")
    print("=" * 60)

    # Print summary
    print("\n📊 QUICK SUMMARY:")
    print(f"Age MAE: {results['age']['MAE']:.2f} years")
    print(f"Gender Accuracy: {results['gender']['accuracy']:.2%}")
    print(f"Ethnicity Accuracy: {results['ethnicity']['accuracy']:.2%}")


if __name__ == "__main__":
    main()
