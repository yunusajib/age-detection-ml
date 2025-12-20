import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =====================================================
# 🔹 Extract Ground Truth Labels
# =====================================================
def extract_targets(dataset):
    """
    Extract true labels from a tf.data.Dataset.
    Automatically handles sparse or one-hot labels.
    """

    y_age, y_gender, y_ethnicity = [], [], []

    for _, labels in dataset:
        # Age (always regression)
        y_age.append(labels["age_output"].numpy())

        # Gender (sparse or one-hot)
        gender = labels["gender_output"].numpy()
        if gender.ndim > 1:
            gender = np.argmax(gender, axis=1)
        y_gender.append(gender)

        # Ethnicity (sparse or one-hot)
        ethnicity = labels["ethnicity_output"].numpy()
        if ethnicity.ndim > 1:
            ethnicity = np.argmax(ethnicity, axis=1)
        y_ethnicity.append(ethnicity)

    return (
        np.concatenate(y_age).reshape(-1),
        np.concatenate(y_gender).reshape(-1),
        np.concatenate(y_ethnicity).reshape(-1),
    )


# =====================================================
# 🔹 Run Model Predictions
# =====================================================
def predict_model(model, dataset):
    """
    Run predictions and normalize outputs.
    Handles dict or list model outputs.
    """

    preds = model.predict(dataset, verbose=0)

    # Dict output (recommended)
    if isinstance(preds, dict):
        pred_age = preds["age_output"].reshape(-1)
        pred_gender = np.argmax(preds["gender_output"], axis=1)
        pred_ethnicity = np.argmax(preds["ethnicity_output"], axis=1)

    # List / tuple output (fallback)
    else:
        pred_age = preds[0].reshape(-1)
        pred_gender = np.argmax(preds[1], axis=1)
        pred_ethnicity = np.argmax(preds[2], axis=1)

    return pred_age, pred_gender, pred_ethnicity


# =====================================================
# 🔹 Metrics
# =====================================================
def evaluate_regression(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
    }


def evaluate_gender(y_true, y_pred):
    """
    Binary classification metrics for gender.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def evaluate_ethnicity(y_true, y_pred):
    """
    Multi-class classification metrics for ethnicity.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# =====================================================
# 🔹 Main Evaluation Function
# =====================================================
def evaluate_model(model, test_ds):
    test_ds = test_ds.cache()

    y_age, y_gender, y_ethnicity = extract_targets(test_ds)
    pred_age, pred_gender, pred_ethnicity = predict_model(model, test_ds)

    n = min(len(y_age), len(pred_age))
    y_age, pred_age = y_age[:n], pred_age[:n]
    y_gender, pred_gender = y_gender[:n], pred_gender[:n]
    y_ethnicity, pred_ethnicity = y_ethnicity[:n], pred_ethnicity[:n]

    return {
        "age": evaluate_regression(y_age, pred_age),
        "gender": evaluate_gender(y_gender, pred_gender),
        "ethnicity": evaluate_ethnicity(y_ethnicity, pred_ethnicity),
        "y_true": {
            "age": y_age,
            "gender": y_gender,
            "ethnicity": y_ethnicity,
        },
        "y_pred": {
            "age": pred_age,
            "gender": pred_gender,
            "ethnicity": pred_ethnicity,
        },
    }
