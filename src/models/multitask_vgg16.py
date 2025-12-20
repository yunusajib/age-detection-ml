"""
Multitask VGG-style CNN for Age, Gender, Ethnicity Prediction
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_multitask_vgg16(input_shape=(48, 48, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    age = layers.Dense(1, name="age_output")(x)
    gender = layers.Dense(2, activation="softmax", name="gender_output")(x)
    ethnicity = layers.Dense(5, activation="softmax",
                             name="ethnicity_output")(x)

    return tf.keras.Model(inputs, [age, gender, ethnicity], name="VGG_Style_Multitask")
