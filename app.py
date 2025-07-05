import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# =====================================
# Title
# =====================================
st.title("Pneumonia Detection using CNN + LSTM")
st.markdown("Upload chest X-ray image or view model evaluation results.")

# =====================================
# Load and preprocess dataset (optional - show results)
# =====================================
@st.cache_data
def load_test_data():
    test_normal = glob.glob(r"D:\Downloads\pnemonia_extract\Pediatric Chest X-ray Pneumonia\test\NORMAL\*.*")
    test_pneumonia = glob.glob(r"D:\Downloads\pnemonia_extract\Pediatric Chest X-ray Pneumonia\test\PNEUMONIA\*.*")

    test_images, test_labels = [], []

    for path in test_normal:
        img = load_img(path, color_mode='rgb', target_size=(224, 224))
        test_images.append(img_to_array(img))
        test_labels.append(0)

    for path in test_pneumonia:
        img = load_img(path, color_mode='rgb', target_size=(224, 224))
        test_images.append(img_to_array(img))
        test_labels.append(1)

    test_x = np.array(test_images, dtype='float32') / 255.0
    test_y = np.array(test_labels)

    return test_x, test_y

# =====================================
# Load model
# =====================================
@st.cache_resource
def load_trained_model():
    return load_model("pneumonia_model.h5")

model = load_trained_model()


# =====================================
# Upload & Predict Section
# =====================================
st.subheader("📷 Upload Chest X-ray Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file, color_mode='rgb', target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = 'Normal' if class_idx == 0 else 'Pneumonia'
    confidence = prediction[0][class_idx]

    st.success(f"Prediction: **{class_label}** ({confidence * 100:.2f}% confidence)")

