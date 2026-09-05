import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

# =========================================================
# Project Config
# =========================================================
PROJECT_NAME = "Multi-Class Image Classification CNN"
DATASET_NAME = "CIFAR-10"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
DEVELOPER_NAME = "Zikra"
TECHNOLOGY = "Python, TensorFlow / Keras, Streamlit"
MODEL_PATH = "cnn_cifar10_model.h5"

st.set_page_config(page_title=PROJECT_NAME, page_icon="🧠", layout="wide")

# =========================================================
# Session State
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# =========================================================
# Theme Styling
# =========================================================
def apply_theme():
    if st.session_state.theme == "Light":
        bg, text, card, border = "#FFFFFF", "#1A1A1A", "#F5F5F7", "#DDDDDD"
    else:
        bg, text, card, border = "#0E1117", "#F5F5F5", "#1C1F26", "#333333"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {text};
        }}
        .info-card {{
            background-color: {card};
            border: 1px solid {border};
            padding: 1.2rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            line-height: 1.8;
        }}
        .footer {{
            text-align: center;
            padding: 1.5rem 0 0.5rem 0;
            margin-top: 2rem;
            border-top: 1px solid {border};
            color: {text};
            font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_theme()

# =========================================================
# Load Model
# =========================================================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_model()

# =========================================================
# Header
# =========================================================
st.title(f"🧠 {PROJECT_NAME}")
st.caption(f"Dataset: {DATASET_NAME} | Developed by {DEVELOPER_NAME}")

# =========================================================
# Sidebar — Settings + Model Status + Project Info
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    theme_choice = st.radio(
        "Theme Toggle",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.divider()
    st.header("📊 Model Status")
    if model is not None:
        st.success("Model loaded successfully!")
        st.write(f"**Input shape:** {model.input_shape}")
        st.write(f"**Output classes:** {model.output_shape[-1]}")
        st.write(f"**Total parameters:** {model.count_params():,}")
    else:
        st.error("Model not found")
        st.caption(f"Run `train_model.py` first to generate `{MODEL_PATH}`.")

    st.divider()
    st.header("ℹ️ Project Information")
    st.markdown(
        f"""
        <div class="info-card">
        <b>Project Name:</b><br>{PROJECT_NAME}<br><br>
        <b>Dataset:</b> {DATASET_NAME}<br><br>
        <b>Classes:</b><br>{', '.join(c.capitalize() for c in CLASS_NAMES)}<br><br>
        <b>Developer:</b> {DEVELOPER_NAME}<br><br>
        <b>Technology:</b><br>{TECHNOLOGY}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Main — Upload + Prediction
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a clear image from 10 classes", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.info("No image uploaded yet.")

with col2:
    st.subheader("🔍 Prediction")
    if uploaded_file is not None:
        if model is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Maintain aspect ratio before downsampling
            img_resized = ImageOps.fit(image, (32, 32), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)[0]
            top_idx = int(np.argmax(preds))

            st.metric(
                "Predicted Class",
                CLASS_NAMES[top_idx].capitalize(),
                f"{preds[top_idx] * 100:.2f}% confidence",
            )

            st.write("**Top Class Probabilities:**")
            for name, prob in sorted(zip(CLASS_NAMES, preds), key=lambda x: -x[1]):
                st.progress(float(prob), text=f"{name.capitalize()}: {prob * 100:.2f}%")
        else:
            st.warning("Model not loaded — train the model first.")
    else:
        st.info("Upload an image to get a prediction.")

st.divider()

# =========================================================
# Footer
# =========================================================
st.markdown(
    f"""
    <div class="footer">
    {PROJECT_NAME}<br>
    Developed by {DEVELOPER_NAME}<br>
    2026
    </div>
    """,
    unsafe_allow_html=True,
)