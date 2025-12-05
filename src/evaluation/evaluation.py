import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def extract_targets(dataset):
    """Extract true labels from a tf.data dataset."""
    y_age, y_gender, y_ethnicity = [], [], []

    for batch in dataset:
        _, labels = batch
        # FIX: Use correct label keys matching your dataset
        y_age.append(labels["age_output"].numpy())
        y_gender.append(labels["gender_output"].numpy())
        y_ethnicity.append(labels["ethnicity_output"].numpy())

    return (
        np.concatenate(y_age),
        np.concatenate(y_gender),
        np.concatenate(y_ethnicity)
    )


def predict_model(model, dataset):
    """Run model predictions."""
    preds = model.predict(dataset)

    # FIX: Handle the dictionary output from your model
    # Your model returns a dict with keys: age_output, gender_output, ethnicity_output
    if isinstance(preds, dict):
        pred_age = preds["age_output"].reshape(-1)
        pred_gender = np.argmax(preds["gender_output"], axis=1)
        pred_ethnicity = np.argmax(preds["ethnicity_output"], axis=1)
    else:
        # Fallback if model returns list/tuple
        pred_age = preds[0].reshape(-1)
        pred_gender = np.argmax(preds[1], axis=1)
        pred_ethnicity = np.argmax(preds[2], axis=1)

    return pred_age, pred_gender, pred_ethnicity


def evaluate_regression(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def evaluate_classification(y_true, y_pred):
    # FIX: Add zero_division parameter to handle edge cases
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


def evaluate_model(model, test_ds):
    """
    Evaluate model on test dataset.

    Returns:
        dict: Contains metrics for age, gender, ethnicity and true/predicted values
    """
    # Extract actual labels
    y_age, y_gender, y_ethnicity = extract_targets(test_ds)

    # Predictions
    pred_age, pred_gender, pred_ethnicity = predict_model(model, test_ds)

    # Compute metrics
    age_metrics = evaluate_regression(y_age, pred_age)
    gender_metrics = evaluate_classification(y_gender, pred_gender)
    ethnicity_metrics = evaluate_classification(y_ethnicity, pred_ethnicity)

    return {
        "age": age_metrics,
        "gender": gender_metrics,
        "ethnicity": ethnicity_metrics,
        "y_true": (y_age, y_gender, y_ethnicity),
        "y_pred": (pred_age, pred_gender, pred_ethnicity)
    }
