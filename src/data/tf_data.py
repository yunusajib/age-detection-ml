import tensorflow as tf
import pandas as pd
import numpy as np
from src.data.preprocess import preprocess_image

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TRAIN_CSV = os.path.join(BASE_DIR, "data/processed/train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data/processed/val.csv")
TEST_CSV = os.path.join(BASE_DIR, "data/processed/test.csv")


def make_dataset(df, batch_size=32, shuffle=True):
    """
    Create a tf.data.Dataset from a cleaned DataFrame.
    """

    # --------------------------------------------
    # 1. CLEAN THE DATAFRAME (THE REAL FIX)
    # --------------------------------------------
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["ethnicity"] = pd.to_numeric(df["ethnicity"], errors="coerce")
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
    df = df.dropna(subset=["age", "ethnicity", "gender"])

    # --------------------------------------------
    # 2. EXTRACT CLEANED NUMERIC ARRAYS
    # --------------------------------------------
    pixel_strings = df["pixels"].astype(str).values
    ages = df["age"].astype(np.float32).values
    ethnicity = df["ethnicity"].astype(np.int32).values
    genders = df["gender"].astype(np.int32).values

    # --------------------------------------------
    # 3. Make dataset
    # --------------------------------------------
    dataset = tf.data.Dataset.from_tensor_slices(
        (pixel_strings, ages, ethnicity, genders)
    )

    # --------------------------------------------
    # 4. Convert string pixels → image tensors
    # --------------------------------------------
    def _process(pixel_string, age, eth, gender):

        pixel_string = pixel_string.numpy().decode("utf-8")
        img = preprocess_image(pixel_string)

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
            "age_output": age,        # Changed from "age" to "age_output"
            "ethnicity_output": eth,  # Changed from "ethnicity" to "ethnicity_output"
            "gender_output": gender   # Changed from "gender" to "gender_output"
        }

        return img, labels

    dataset = dataset.map(_tf_process, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(1024)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def load_train_data(batch_size=32):
    """Load training dataset from CSV."""
    df = pd.read_csv(TRAIN_CSV)
    return make_dataset(df, batch_size=batch_size, shuffle=True)


def load_val_data(batch_size=32):
    """Load validation dataset from CSV."""
    df = pd.read_csv(VAL_CSV)
    return make_dataset(df, batch_size=batch_size, shuffle=False)


def load_test_data(batch_size=32):
    """Load test dataset from CSV."""
    df = pd.read_csv(TEST_CSV)
    return make_dataset(df, batch_size=batch_size, shuffle=False)
