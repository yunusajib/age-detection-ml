import tensorflow as tf
import pandas as pd
from src.data.preprocess import preprocess_image


def load_split_csv(csv_path):
    return pd.read_csv(csv_path)


def load_image_from_pixels(pixel_string):
    img = preprocess_image(pixel_string)
    return tf.convert_to_tensor(img, dtype=tf.float32)


def make_dataset(df, batch_size=32, shuffle=True):
    pixel_strings = df["pixels"].values
    ages = df["age"].values
    ethnicity = df["ethnicity"].values
    genders = df["gender"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (pixel_strings, ages, ethnicity, genders)
    )

    # Python-side processing
    def _process(pixel_string, age, eth, gender):
        img = load_image_from_pixels(pixel_string.numpy().decode("utf-8"))
        return (
            img,
            tf.cast(age, tf.float32),
            tf.cast(eth, tf.int32),
            tf.cast(gender, tf.int32),
        )

    # TensorFlow wrapper
    def _tf_process(pixel_string, age, eth, gender):
        img, age_out, eth_out, gender_out = tf.py_function(
            func=_process,
            inp=[pixel_string, age, eth, gender],
            Tout=[tf.float32, tf.float32, tf.int32, tf.int32],
        )

        img.set_shape((48, 48, 3))
        age_out.set_shape(())
        eth_out.set_shape(())
        gender_out.set_shape(())

        labels = {
            "age": age_out,
            "ethnicity": eth_out,
            "gender": gender_out,
        }

        return img, labels

    # Apply mapping
    dataset = dataset.map(_tf_process, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
