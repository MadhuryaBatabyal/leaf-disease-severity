import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('tomato_blight.keras')

# Rest same as before...


model = load_model()

severity_map = {
    0: ('Healthy', 0, '🟢 No infection'),
    1: ('Mild', 12, '🟡 Low spread: 0-25%'),
    2: ('Moderate', 37, '🟠 Medium spread: 25-50%'),
    3: ('Severe', 75, '🔴 High spread: 50-100%')
}

st.title("🍅 Tomato Late Blight Severity Detector")
uploaded = st.file_uploader("Upload tomato leaf...", type=['jpg','png','jpeg'])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB').resize((224,224))
    st.image(img, caption="Uploaded leaf", use_column_width=True)
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)[0]
    class_id = np.argmax(pred)
    confidence = pred[class_id] * 100
    
    label, percent, desc = severity_map[class_id]
    
    st.subheader(f"**{label}** ({confidence:.1f}% confidence)")
    col1, col2 = st.columns(2)
    col1.metric("Infection Spread", f"{percent}%")
    col2.progress(percent / 100)
    st.info(desc)
