# app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import os

# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models",
                          "vgg16_48_best.keras")  # or any model you want
CLASS_LABELS = {
    "gender": ["Male", "Female"],
    "ethnicity": ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
}


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model(path):
    return keras.models.load_model(path, compile=False)


model = load_model(MODEL_PATH)


# -----------------------------
# Preprocess uploaded image
# -----------------------------
def preprocess_uploaded_image(img: Image.Image):
    img = img.convert("RGB")            # ensure 3 channels
    img = img.resize((48, 48))          # resize to 48x48
    img_array = np.array(img).astype(np.float32)
    img_array /= 255.0                  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


# -----------------------------
# Prediction
# -----------------------------
def predict(model, img: Image.Image):
    img_array = preprocess_uploaded_image(img)
    preds = model.predict(img_array, verbose=0)

    if isinstance(preds, dict):
        age_pred = preds["age_output"][0][0]
        gender_pred = np.argmax(preds["gender_output"][0])
        ethnicity_pred = np.argmax(preds["ethnicity_output"][0])
    else:
        age_pred = preds[0][0][0]
        gender_pred = np.argmax(preds[1][0])
        ethnicity_pred = np.argmax(preds[2][0])

    return age_pred, gender_pred, ethnicity_pred


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Age, Gender & Ethnicity Prediction")
st.write("Upload a face image, and the model will predict age, gender, and ethnicity.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        age, gender, ethnicity = predict(model, image)

    st.success("Prediction Complete!")
    st.write(f"**Predicted Age:** {age:.1f} years")
    st.write(f"**Predicted Gender:** {CLASS_LABELS['gender'][gender]}")
    st.write(
        f"**Predicted Ethnicity:** {CLASS_LABELS['ethnicity'][ethnicity]}")
