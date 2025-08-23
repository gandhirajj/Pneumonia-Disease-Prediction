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

# Function to validate if the uploaded image looks like a chest X-ray
def is_valid_xray(pil_img):
    try:
        img = np.array(pil_img)

        # Step 1: Check if grayscale (true X-rays are grayscale)
        if len(img.shape) == 2:  # Already grayscale
            img_gray = img
        elif len(img.shape) == 3 and img.shape[2] == 3:
            diff_rg = np.abs(img[:,:,0] - img[:,:,1]).mean()
            diff_rb = np.abs(img[:,:,0] - img[:,:,2]).mean()
            diff_gb = np.abs(img[:,:,1] - img[:,:,2]).mean()
            if not (diff_rg < 2 and diff_rb < 2 and diff_gb < 2):
                return False
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            return False

        # Step 2: Intensity distribution check
        mean_intensity = np.mean(img_gray)
        if mean_intensity > 230 or mean_intensity < 40:
            return False

        # Step 3: Edge density check
        img_resized = cv2.resize(img_gray, (224, 224))
        edges = cv2.Canny(img_resized, 50, 150)
        edge_density = np.sum(edges) / edges.size
        if edge_density < 0.01:  # empirically chosen
            return False

        return True
    except Exception as e:
        print("Validation Error:", e)
        return False

# When image is uploaded
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Step 1: Is image a valid chest X-ray?
    if not is_valid_xray(img):
        st.error("❌ This image is not a valid chest X-ray. Please upload a proper grayscale X-ray.")
    else:
        img = img.convert("RGB")  # Convert safely for model
        st.image(img, caption="Valid Chest X-ray", use_column_width=True)

        # Step 2: Preprocess for model
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Step 3: Predict
        prediction = model.predict(img_array)[0][0]
        label = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        st.success(f"✅ Prediction: {label} ({confidence * 100:.2f}% confidence)")
