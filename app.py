# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
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
# Load model
# -----------------------------
@st.cache_resource
def load_model(path):
    """Load Keras 3.x model"""
    try:
        return keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"TensorFlow version: {tf.__version__}")
        raise

# Load model
model = load_model(MODEL_PATH)
st.success(f"✅ Model loaded successfully! (TensorFlow {tf.__version__})")

# -----------------------------
# Preprocess uploaded image
# -----------------------------
def preprocess_uploaded_image(img: Image.Image):
    """Convert image to model input format"""
    img = img.convert("RGB")
    img = img.resize((48, 48))
    img_array = np.array(img).astype(np.float32)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Prediction
# -----------------------------
def predict(model, img: Image.Image):
    """Run prediction on uploaded image"""
    img_array = preprocess_uploaded_image(img)
    preds = model.predict(img_array, verbose=0)
    
    # Handle different output formats
    if isinstance(preds, dict):
        age_pred = preds["age_output"][0][0]
        gender_pred = np.argmax(preds["gender_output"][0])
        ethnicity_pred = np.argmax(preds["ethnicity_output"][0])
    else:
        # preds is a list of arrays
        age_pred = preds[0][0][0]
        gender_pred = np.argmax(preds[1][0])
        ethnicity_pred = np.argmax(preds[2][0])
    
    return age_pred, gender_pred, ethnicity_pred

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Age Detection ML", page_icon="🎭")

st.title("🎭 Age, Gender & Ethnicity Prediction")
st.write("Upload a face image, and the AI model will predict age, gender, and ethnicity.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear face image for best results"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    with st.spinner("🔮 Analyzing image..."):
        try:
            age, gender, ethnicity = predict(model, image)
            
            st.success("✨ Prediction Complete!")
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📅 Predicted Age",
                    value=f"{age:.1f} years"
                )
            
            with col2:
                st.metric(
                    label="👤 Predicted Gender",
                    value=CLASS_LABELS['gender'][gender]
                )
            
            with col3:
                st.metric(
                    label="🌍 Predicted Ethnicity",
                    value=CLASS_LABELS['ethnicity'][ethnicity]
                )
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Footer
st.divider()
st.caption("Built with Streamlit and TensorFlow | VGG16 Model")
