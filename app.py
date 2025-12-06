"""
Streamlit demo for Facial Attribute Prediction
Simpler deployment than Gradio/HF
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Facial Attribute Prediction",
    page_icon="🎭",
    layout="wide"
)

# Title
st.title("🎭 Multi-Task Facial Attribute Prediction")
st.markdown("Upload a facial image to predict **age**, **gender**, and **ethnicity**")

# Sidebar
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Age MAE", "±7.6 years")
    st.metric("Gender Accuracy", "71.2%")
    st.metric("Ethnicity Accuracy", "54.7%")
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repo](https://github.com/yunusajib/age-detection-ml)")
    st.markdown("[Model Details](https://github.com/yunusajib/age-detection-ml#readme)")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a face image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🔮 Predictions")
    
    if uploaded_file is not None:
        # Load model lazily (only when needed)
        @st.cache_resource
        def load_model():
            try:
                from tensorflow import keras
                import tensorflow as tf
                
                # Try loading without custom objects first
                try:
                    model = keras.models.load_model('models/model_deployment.keras')
                    return model, None
                except:
                    # Fallback: load with custom objects
                    class WeightedSparseCategoricalCrossentropy(keras.losses.Loss):
                        def __init__(self, class_weights=None, **kwargs):
                            super().__init__(**kwargs)
                            self.class_weights = class_weights or {}
                        
                        def call(self, y_true, y_pred):
                            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                            return loss_fn(y_true, y_pred)
                        
                        def get_config(self):
                            config = super().get_config()
                            config.update({"class_weights": self.class_weights})
                            return config
                    
                    model = keras.models.load_model(
                        'models/best_model_improved.keras',
                        custom_objects={'WeightedSparseCategoricalCrossentropy': WeightedSparseCategoricalCrossentropy},
                        compile=False
                    )
                    return model, None
            except Exception as e:
                return None, str(e)
        
        with st.spinner("Loading model..."):
            model, error = load_model()
        
        if error:
            st.error(f"Error loading model: {error}")
            st.info("💡 For demo purposes, showing sample predictions:")
            
            # Demo mode - show fake predictions
            st.success("✅ Demo Mode Active")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Age", "28 years")
            with col_b:
                st.metric("Gender", "Female", delta="72% confidence")
            with col_c:
                st.metric("Ethnicity", "Class 1", delta="65% confidence")
            
            st.warning("⚠️ Note: These are sample predictions for demonstration. Deploy with actual model for real predictions.")
            
        else:
            # Preprocess image
            def preprocess_image(img):
                img_array = np.array(img)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, (48, 48))
                rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                normalized = rgb.astype(np.float32) / 255.0
                return np.expand_dims(normalized, axis=0)
            
            processed = preprocess_image(image)
            
            # Predict
            with st.spinner("Making predictions..."):
                predictions = model.predict(processed, verbose=0)
            
            # Parse predictions
            if isinstance(predictions, dict):
                age = predictions["age_output"][0][0]
                gender = predictions["gender_output"][0]
                ethnicity = predictions["ethnicity_output"][0]
            else:
                age = predictions[0][0][0]
                gender = predictions[1][0]
                ethnicity = predictions[2][0]
            
            # Display results
            st.success("✅ Predictions Complete!")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Predicted Age", f"{int(age)} years")
            
            with col_b:
                gender_label = "Male" if gender[0] > gender[1] else "Female"
                confidence = max(gender[0], gender[1]) * 100
                st.metric("Gender", gender_label, delta=f"{confidence:.1f}% confidence")
            
            with col_c:
                eth_class = np.argmax(ethnicity)
                eth_confidence = ethnicity[eth_class] * 100
                st.metric("Ethnicity", f"Class {eth_class}", delta=f"{eth_confidence:.1f}% confidence")
            
            # Detailed probabilities
            with st.expander("📊 Detailed Probabilities"):
                st.markdown("**Gender Probabilities:**")
                st.progress(float(gender[0]), text=f"Male: {gender[0]*100:.1f}%")
                st.progress(float(gender[1]), text=f"Female: {gender[1]*100:.1f}%")
                
                st.markdown("**Ethnicity Probabilities:**")
                for i in range(5):
                    st.progress(float(ethnicity[i]), text=f"Class {i}: {ethnicity[i]*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
### ℹ️ About This Model
- **Architecture**: Multi-task CNN with shared feature extraction
- **Training**: 18,964 samples from UTKFace dataset
- **Improvements**: +27.5% ethnicity accuracy, +16.1% gender accuracy over baseline
- **Regularization**: Dropout (0.5), data augmentation, class-weighted losses

Built with ❤️ using TensorFlow/Keras and Streamlit
""")