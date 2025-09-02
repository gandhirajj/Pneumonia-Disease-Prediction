import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import time
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -----------------------------
# Import CrewAI for recommendations
# -----------------------------
from crewai import Agent, Task, Crew

# ======================
# App Title & Sidebar
# ======================
st.set_page_config(page_title="Pneumonia Detector with AI Recommendations", page_icon="🩻", layout="wide")
st.title("Pneumonia Detection Dashboard with AI Recommendations")
st.sidebar.header("📁 Patient Metadata")

# ------------------
# Patient Metadata Inputs
# ------------------
age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)
gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"])
symptoms = st.sidebar.multiselect("Symptoms", ["Cough", "Fever", "Chest Pain", "Fatigue"])
smoking_history = st.sidebar.selectbox("Smoking History", ["Non-smoker", "Former smoker", "Current smoker"])
location = st.sidebar.text_input("City or Zip Code")

# ------------------
# Load Model
# ------------------
@st.cache_resource
def load_main_model():
    return load_model("best_model_mobilenet_lstm.h5")  # replace with your path

model = load_main_model()
model_version = "v1.0"

# ------------------
# Image Validation & Preprocessing
# ------------------
def is_valid_xray(pil_img):
    try:
        img = np.array(pil_img)
        if len(img.shape) == 2:
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
        mean_intensity = np.mean(img_gray)
        if mean_intensity > 230 or mean_intensity < 40:
            return False
        img_resized = cv2.resize(img_gray, (224, 224))
        edges = cv2.Canny(img_resized, 50, 150)
        edge_density = np.sum(edges) / edges.size
        if edge_density < 0.01:
            return False
        return True
    except:
        return False

def preprocess(img):
    img = img.convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)/255.0
    return np.expand_dims(img_array, axis=0)

# ------------------
# Grad-CAM
# ------------------
def generate_gradcam(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)/np.max(heatmap)
    heatmap = cv2.resize(heatmap, (224,224))
    return heatmap

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    img = np.array(original_img.convert("RGB").resize((224,224)))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return overlay

# ------------------
# PDF Export
# ------------------
def export_pdf(report_text, img, heatmap_img):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    text_obj = c.beginText(40, height-40)
    for line in report_text.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)
    if img is not None:
        img_rgb = img.convert("RGB").resize((200,200))
        img_rgb.save("xray_temp.png")
        c.drawImage("xray_temp.png", 40, height//2 - 100, width=200, preserveAspectRatio=True)
    if heatmap_img is not None:
        heatmap_pil = Image.fromarray(heatmap_img)
        heatmap_pil.save("heatmap_temp.png")
        c.drawImage("heatmap_temp.png", 280, height//2 - 100, width=200, preserveAspectRatio=True)
    c.save()
    buffer.seek(0)
    return buffer

# ------------------
# CrewAI Agents
# ------------------
class HospitalRecommendationAgent(Agent):
    def run(self, location_input):
        # Demo static hospital recommendation
        return [
            {"name": "City Hospital", "address": f"{location_input}, Main Street", "rating": 4.5},
            {"name": "Pulmonology Center", "address": f"{location_input}, 2nd Avenue", "rating": 4.2},
            {"name": "General Hospital", "address": f"{location_input}, 3rd Road", "rating": 4.0},
        ]

# ------------------
# File Upload & Prediction
# ------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["png","jpg","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    if not is_valid_xray(img):
        st.error("❌ This is not a valid chest X-ray. Please upload a proper scan.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Chest X-ray", use_column_width=True)

        # Prediction
        start_time = time.time()
        img_array = preprocess(img)
        prediction = model.predict(img_array)[0][0]
        inference_time = time.time()-start_time
        label = "Pneumonia" if prediction>0.5 else "Normal"
        confidence = prediction if prediction>0.5 else 1-prediction
        with col2:
            st.metric("Prediction", label)
            st.metric("Confidence", f"{confidence*100:.2f}%")
            st.metric("Inference Time", f"{inference_time:.2f} sec")
            st.metric("Model Version", model_version)

        # Grad-CAM
        heatmap = generate_gradcam(model, img_array)
        overlay = overlay_heatmap(img, heatmap)
        st.subheader("🔍 Visual Explanation (Grad-CAM)")
        col3, col4 = st.columns(2)
        with col3:
            st.image(img, caption="Original X-ray", use_column_width=True)
        with col4:
            st.image(overlay, caption="Highlighted Regions", use_column_width=True)

        # Recommendations
        st.subheader("🧪 Recommendations & Next Steps")
        if label == "Pneumonia":
            st.error("⚠️ Pneumonia detected. Please consult a doctor immediately.")
            if location:
                # CrewAI hospital recommendation
                agent = HospitalRecommendationAgent(name="Hospital Recommender")
                hospitals = agent.run(location)
                st.subheader("🏥 Recommended Hospitals Near You")
                for h in hospitals:
                    st.markdown(f"- **{h['name']}**, {h['address']}, Rating: {h['rating']}/5")
        else:
            st.success("✅ No pneumonia detected. Continue regular monitoring if symptoms persist.")

        if symptoms:
            st.info(f"📌 Reported symptoms: {', '.join(symptoms)}")

        # PDF Export
        st.subheader("📤 Export Report")
        report_text = f"""
Patient Report
-------------------------
Age: {age}
Gender: {gender}
Smoking History: {smoking_history}
Symptoms: {', '.join(symptoms) if symptoms else 'None'}

Prediction: {label}
Confidence: {confidence*100:.2f}%
Inference Time: {inference_time:.2f} sec
Model Version: {model_version}
"""
        if label=="Pneumonia" and location:
            hospital_list = "\n".join([f"{h['name']} - {h['address']} (Rating: {h['rating']}/5)" for h in hospitals])
            report_text += f"\nRecommended Hospitals:\n{hospital_list}"

        pdf_buffer = export_pdf(report_text, img, overlay)
        st.download_button(
            label="📥 Download Report (PDF)",
            data=pdf_buffer,
            file_name="pneumonia_report.pdf",
            mime="application/pdf"
        )
