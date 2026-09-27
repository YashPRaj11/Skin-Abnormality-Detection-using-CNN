import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import os
import gdown

MODEL_PATH = "skin.keras"
# Load model
model = load_model(MODEL_PATH)

img_size = (224, 224)

st.title("🧠 Skin Abnormality Detection")
st.write("Upload an image to detect skin abnormality")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def preprocess_image(image):
    image = image.resize(img_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    st.write(prediction)
    if prediction < 0.5:
        st.error(f"⚠️ Abnormal Skin Detected ({prediction:.2f})")
    else:
        st.success(f"✅ Normal Skin ({1 - prediction:.2f})")