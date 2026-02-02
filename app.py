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
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "vgg16_48_best.keras")

CLASS_LABELS = {
    "gender": ["Male", "Female"],
    "ethnicity": ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
}

# -----------------------------
# Load model with better error handling
# -----------------------------
@st.cache_resource
def load_model(path):
    try:
        # Try loading with safe_mode=False (for newer Keras versions)
        return keras.models.load_model(path, compile=False, safe_mode=False)
    except Exception as e:
        st.error(f"Error loading model with safe_mode=False: {e}")
        try:
            # Fallback: try without safe_mode parameter
            return keras.models.load_model(path, compile=False)
        except Exception as e2:
            st.error(f"Error loading model: {e2}")
            st.error(f"Model path: {path}")
            st.error(f"Model exists: {os.path.exists(path)}")
            raise

# Load model
try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model. Please check the logs.")
    st.stop()

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
    
    # Handle different output formats
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
st.title("🎭 Age, Gender & Ethnicity Prediction")
st.write("Upload a face image, and the model will predict age, gender, and ethnicity.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Predicting..."):
        try:
            age, gender, ethnicity = predict(model, image)
            
            st.success("Prediction Complete!")
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Age", f"{age:.1f} years")
            
            with col2:
                st.metric("Predicted Gender", CLASS_LABELS['gender'][gender])
            
            with col3:
                st.metric("Predicted Ethnicity", CLASS_LABELS['ethnicity'][ethnicity])
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
