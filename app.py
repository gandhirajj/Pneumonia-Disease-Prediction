import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from keras.models import load_model

st.title("🩻 Pneumonia Detection from Chest X-ray")
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])

# Load model once and cache it
@st.cache_resource
def load_main_model():
    return load_model("best_model_mobilenet_lstm.h5")  # Your trained pneumonia model

model = load_main_model()

# Function to check if image looks like a chest X-ray
def is_valid_xray(pil_img):
    try:
        img = np.array(pil_img.resize((224, 224)))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Check 1: Grayscale-like intensity distribution
        if np.mean(img_gray) > 230 or np.mean(img_gray) < 40:
            return False

        # Check 2: Edge density (chest X-rays have moderate structural edges)
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        if edge_density < 5:  # empirically low for irrelevant images
            return False

        return True
    except Exception as e:
        print("CV Error:", e)
        return False

# When image is uploaded
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Step 1: Is image likely to be a chest X-ray?
    if not is_valid_xray(img):
        st.warning("⚠️ This image doesn't appear to be a valid chest X-ray.")
    else:
        st.image(img, caption="Input Image", use_column_width=True)

        # Step 2: Preprocess for model
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Step 3: Predict
        prediction = model.predict(img_array)[0][0]
        label = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        st.success(f"✅ Prediction: {label} ({confidence * 100:.2f}% confidence)")
