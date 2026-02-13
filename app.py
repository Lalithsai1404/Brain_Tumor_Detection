# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/brain_tumor_model.h5')

model = load_model()

st.title("🧠 Multi-class Brain Tumor Detection")
uploaded = st.file_uploader("Upload MRI Image", type=['jpg','jpeg','png'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded MRI', use_container_width=True)

    img_arr = np.array(img.resize((224,224))) / 255.0
    pred = model.predict(np.expand_dims(img_arr, axis=0))
    class_idx = np.argmax(pred)
    class_labels = list(os.listdir('data/Training'))
    st.write(f"Prediction: {class_labels[class_idx]}")
    st.write(f"Confidence: {pred[0][class_idx]:.4f}")
