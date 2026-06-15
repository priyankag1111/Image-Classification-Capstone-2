import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Clothing Classification App")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("clothing_classifier.keras", compile=False)

model = load_model("clothing_classifier.keras")

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
