import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('tomato_blight_severity.h5')

model = load_model()

# Severity mapping
severity_map = {
    0: ('Healthy', 0, '🟢 No infection'),
    1: ('Mild', 12, '🟡 Low spread: 0-25%'),
    2: ('Moderate', 37, '🟠 Medium spread: 25-50%'),
    3: ('Severe', 75, '🔴 High spread: 50-100%')
}

st.title("🍅 Tomato Late Blight Severity Detector")
st.write("Upload a tomato leaf image to check disease spread %.")

uploaded = st.file_uploader("Choose image...", type=['jpg', 'jpeg', 'png'])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded", use_column_width=True)
    
    # Preprocess
    img_array = np.array(img.resize((224,224)))
    img_array = np.expand_dims(img_array / 255.0, axis=0)
    
    # Predict
    pred = model.predict(img_array)[0]
    class_id = np.argmax(pred)
    confidence = pred[class_id] * 100
    
    label, percent, desc = severity_map[class_id]
    
    st.subheader(f"**Prediction: {label} ({confidence:.1f}% confidence)**")
    st.metric("Est. Infection Spread", f"{percent}%", delta=None)
    st.progress(percent / 100)
    st.info(desc)
    
    if class_id > 0:
        st.warning("🚨 Disease detected! Monitor/treat early.")
