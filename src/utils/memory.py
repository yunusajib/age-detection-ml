import tensorflow as tf
import gc


def reset_tf():
    """
    Clear TensorFlow/Keras memory.
    Call BEFORE building a new model.
    """
    tf.keras.backend.clear_session()
    gc.collect()
