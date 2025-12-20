"""
Baseline multi-task CNN for age, gender, and ethnicity prediction.
This is the initial architecture without advanced regularization.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_multitask_model(input_shape=(48, 48, 3)):
    """
    Build baseline multi-task CNN model.

    Args:
        input_shape: Input image shape (height, width, channels)

    Returns:
        Keras Model with three outputs: age, gender, ethnicity
    """

    # Input layer
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Convolutional backbone
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten
    x = layers.Flatten()(x)

    # Shared dense layer
    x = layers.Dense(512, activation='relu')(x)

    # ==========================================
    # Task-specific heads
    # ==========================================

    # Age regression head
    age_branch = layers.Dense(128, activation='relu')(x)
    age_output = layers.Dense(1, activation='linear',
                              name='age_output')(age_branch)

    # Gender classification head (binary: 0=male, 1=female)
    gender_branch = layers.Dense(64, activation='relu')(x)
    gender_output = layers.Dense(
        2, activation='softmax', name='gender_output')(gender_branch)

    # Ethnicity classification head (5 classes)
    ethnicity_branch = layers.Dense(128, activation='relu')(x)
    ethnicity_output = layers.Dense(
        5, activation='softmax', name='ethnicity_output')(ethnicity_branch)

    # Build model
    model = keras.Model(
        inputs=inputs,
        outputs=[age_output, gender_output, ethnicity_output],
        name="multitask_cnn_baseline"
    )

    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating baseline multi-task CNN...")
    model = build_multitask_model()

    print("\n" + "="*60)
    model.summary()

    print("\n" + "="*60)
    print("BASELINE MODEL CHARACTERISTICS:")
    print("="*60)
    print("❌ No dropout layers")
    print("❌ No batch normalization")
    print("❌ No regularization")
    print("❌ Prone to overfitting")
    print("❌ Class imbalance not addressed")

    print(f"\n✅ Baseline model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
