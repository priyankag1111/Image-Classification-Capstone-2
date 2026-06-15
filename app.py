import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import os

st.write("Current files:", os.listdir())

try:
    model = tf.keras.models.load_model("clothing_classifier.keras", compile=False)
    st.write("Model loaded successfully")
except Exception as e:
    st.error("Model loading failed")
    st.exception(e)
st.title("Clothing Classification App")

import streamlit as st
import tensorflow as tf

st.write("Loading model...")

model = tf.keras.models.load_model("clothing_classifier.keras")

st.write("Model loaded successfully")

CLASS_NAMES = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt"
]

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Predict"):
        pred = model.predict(preprocess_image(image))
        class_id = np.argmax(pred)
        confidence = np.max(pred)

        st.success(f"Prediction: {CLASS_NAMES[class_id]}")
        st.info(f"Confidence: {confidence:.2%}")
