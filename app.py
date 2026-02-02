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
# Load model with compatibility fixes
# -----------------------------
@st.cache_resource
def load_model(path):
    """Load model with multiple fallback strategies"""
    
    # Strategy 1: Try with safe_mode=False (Keras 3.x compatibility)
    try:
        st.info("Loading model with safe_mode=False...")
        model = keras.models.load_model(path, compile=False, safe_mode=False)
        st.success("✅ Model loaded successfully with safe_mode=False!")
        return model
    except TypeError as e:
        st.warning(f"safe_mode parameter not supported, trying without it...")
    except Exception as e:
        st.warning(f"Failed with safe_mode=False: {str(e)[:100]}")
    
    # Strategy 2: Try legacy H5 format loading
    try:
        st.info("Trying legacy H5 format loading...")
        model = tf.keras.models.load_model(path, compile=False)
        st.success("✅ Model loaded with legacy method!")
        return model
    except Exception as e:
        st.warning(f"Legacy loading failed: {str(e)[:100]}")
    
    # Strategy 3: Try with custom objects (empty dict can help with some compatibility issues)
    try:
        st.info("Trying with custom_objects parameter...")
        model = keras.models.load_model(path, compile=False, custom_objects={})
        st.success("✅ Model loaded with custom_objects!")
        return model
    except Exception as e:
        st.error(f"All loading strategies failed!")
        st.error(f"Model path: {path}")
        st.error(f"Model exists: {os.path.exists(path)}")
        st.error(f"Final error: {e}")
        raise

# Load model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("❌ Critical error: Failed to load model after all attempts.")
    st.error("This usually means the model was saved with a different TensorFlow/Keras version.")
    st.error(f"Current TensorFlow version: {tf.__version__}")
    st.error(f"Current Keras version: {keras.__version__}")
    st.info("💡 Recommendation: Re-save your model with TensorFlow 2.15.0 or update requirements.txt to match the training version.")
    st.stop()

# -----------------------------
# Preprocess uploaded image
# -----------------------------
def preprocess_uploaded_image(img: Image.Image):
    """Convert image to model input format"""
    img = img.convert("RGB")            # ensure 3 channels
    img = img.resize((48, 48))          # resize to 48x48
    img_array = np.array(img).astype(np.float32)
    img_array /= 255.0                  # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# -----------------------------
# Prediction
# -----------------------------
def predict(model, img: Image.Image):
    """Run prediction on uploaded image"""
    img_array = preprocess_uploaded_image(img)
    preds = model.predict(img_array, verbose=0)
    
    # Handle different output formats (dict vs list)
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
            
            # Display results in nice columns
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
            st.error("Please try uploading a different image or contact support.")

# Footer
st.divider()
st.caption("Built with Streamlit and TensorFlow | VGG16 Model")
