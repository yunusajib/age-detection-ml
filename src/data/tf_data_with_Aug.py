import tensorflow as tf
import pandas as pd
import numpy as np
from src.data.preprocess import preprocess_image
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TRAIN_CSV = os.path.join(BASE_DIR, "data/processed/train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data/processed/val.csv")
TEST_CSV = os.path.join(BASE_DIR, "data/processed/test.csv")


def make_dataset(df, batch_size=32, shuffle=True, augment=False):
    """
    Create a tf.data.Dataset from a cleaned DataFrame.

    Args:
        df: DataFrame with pixels, age, ethnicity, gender columns
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation (training only)
    """

    # Clean the dataframe
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["ethnicity"] = pd.to_numeric(df["ethnicity"], errors="coerce")
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
    df = df.dropna(subset=["age", "ethnicity", "gender"])

    # Extract cleaned numeric arrays
    pixel_strings = df["pixels"].astype(str).values
    ages = df["age"].astype(np.float32).values
    ethnicity = df["ethnicity"].astype(np.int32).values
    genders = df["gender"].astype(np.int32).values

    # Make dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (pixel_strings, ages, ethnicity, genders)
    )

    # Convert string pixels to image tensors
    def _process(pixel_string, age, eth, gender):
        pixel_string = pixel_string.numpy().decode("utf-8")
        img = preprocess_image(pixel_string, augment=augment)

        return (img.astype("float32"),
                np.float32(age),
                np.int32(eth),
                np.int32(gender))

    def _tf_process(pixel_string, age, eth, gender):
        img, age, eth, gender = tf.py_function(
            func=_process,
            inp=[pixel_string, age, eth, gender],
            Tout=[tf.float32, tf.float32, tf.int32, tf.int32]
        )

        img.set_shape((48, 48, 3))
        age.set_shape(())
        eth.set_shape(())
        gender.set_shape(())

        labels = {
            "age_output": age,
            "ethnicity_output": eth,
            "gender_output": gender
        }

        return img, labels

    dataset = dataset.map(_tf_process, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(1024)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def load_train_data(batch_size=32, augment=True):
    """Load training dataset with optional augmentation."""
    df = pd.read_csv(TRAIN_CSV)
    return make_dataset(df, batch_size=batch_size, shuffle=True, augment=augment)


def load_val_data(batch_size=32):
    """Load validation dataset (no augmentation)."""
    df = pd.read_csv(VAL_CSV)
    return make_dataset(df, batch_size=batch_size, shuffle=False, augment=False)


def load_test_data(batch_size=32):
    """Load test dataset (no augmentation)."""
    df = pd.read_csv(TEST_CSV)
    return make_dataset(df, batch_size=batch_size, shuffle=False, augment=False)
