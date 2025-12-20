import numpy as np
import tensorflow as tf


def parse_pixels(pixel_string):
    """
    Convert pixel string -> 48x48 numpy array (grayscale).
    """
    pixels = np.fromstring(pixel_string, sep=' ')
    img = pixels.reshape(48, 48)
    return img


def to_3_channels(img):
    """
    Convert grayscale 48x48 -> 48x48x3 by stacking channels.
    """
    return np.stack([img, img, img], axis=-1)


def normalize(img):
    """
    Normalize pixel values to 0–1.
    """
    return img.astype("float32") / 255.0


def preprocess_image(pixel_string):
    """
    Full pipeline:
        1. parse pixels
        2. convert grayscale → 3 channels
        3. normalize
    Returns: (48,48,3)
    """
    img = parse_pixels(pixel_string)
    img = to_3_channels(img)
    img = normalize(img)
    return img
