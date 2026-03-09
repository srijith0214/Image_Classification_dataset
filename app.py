"""
Streamlit Web Application – Fish Image Classifier
Multiclass Fish Image Classification Project

Run:
    streamlit run app.py -- --model_path outputs/best_model.h5
                             --classes_path outputs/class_names.json
"""

import os
import json
import argparse
import sys

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐟 Fish Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title  { font-size: 2.4rem; font-weight: 700; color: #1a6b8a; text-align: center; }
    .subtitle    { font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 1.5rem; }
    .pred-box    { background: #e8f5e9; border-left: 5px solid #388e3c;
                   padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .conf-box    { background: #e3f2fd; border-left: 5px solid #1976d2;
                   padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .metric-card { background: #fff3e0; border-radius: 10px; padding: 1rem;
                   text-align: center; border: 1px solid #ffe0b2; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_class_names(classes_path: str):
    if not os.path.exists(classes_path):
        return []
    with open(classes_path) as f:
        return json.load(f)


def preprocess_image(pil_image: Image.Image, img_size: tuple = (224, 224)) -> np.ndarray:
    """Resize, convert to RGB, normalise to [0,1], add batch dim."""
    img = pil_image.convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def make_confidence_bar(class_names, probs, top_k=5):
    """Return a Plotly horizontal bar chart of top-k predictions."""
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_classes = [class_names[i] for i in top_idx]
    top_probs   = [float(probs[i]) * 100 for i in top_idx]

    colors = ["#1976d2" if i == 0 else "#90caf9" for i in range(len(top_idx))]

    fig = go.Figure(go.Bar(
        x=top_probs,
        y=top_classes,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in top_probs],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 110], title="Confidence (%)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=10, b=30),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/tropical-fish.png", width=80)
    st.title("⚙️ Configuration")

    default_model  = "outputs/best_model.h5"
    default_cls    = "outputs/class_names.json"

    model_path = st.text_input("Model path (.h5)", value=default_model)
    classes_path = st.text_input("Class names JSON", value=default_cls)

    img_size_opt = st.selectbox("Input image size", [224, 299], index=0,
                                help="Use 299 for InceptionV3")
    top_k = st.slider("Top-K predictions to show", 3, 10, 5)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "Upload a fish image and the model will predict its species "
        "along with confidence scores."
    )
    st.markdown("---")
    st.caption("Multiclass Fish Image Classification Project")


# ─── Main Panel ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🐟 Fish Species Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a fish image to identify its species using Deep Learning</p>',
    unsafe_allow_html=True,
)

# ── Load model & class names ──────────────────────────────────────────────────
model = load_model(model_path)
class_names = load_class_names(classes_path)

col_status1, col_status2 = st.columns(2)
with col_status1:
    if model is not None:
        st.success(f"✅ Model loaded: `{os.path.basename(model_path)}`")
    else:
        st.error("❌ Model not found. Train first or check path.")
with col_status2:
    if class_names:
        st.info(f"📋 {len(class_names)} species loaded")
    else:
        st.warning("⚠️ class_names.json not found.")

st.markdown("---")

# ── File Uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📤 Upload a fish image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
)

if uploaded is not None:
    pil_img = Image.open(uploaded)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("📷 Uploaded Image")
        st.image(pil_img, use_column_width=True, caption=uploaded.name)

        # Image metadata
        with st.expander("🔍 Image Details"):
            st.write(f"**Filename:** {uploaded.name}")
            st.write(f"**Size:** {pil_img.size[0]} × {pil_img.size[1]} px")
            st.write(f"**Mode:** {pil_img.mode}")
            st.write(f"**File size:** {uploaded.size / 1024:.1f} KB")

    with right_col:
        st.subheader("🔮 Prediction")

        if model is None or not class_names:
            st.error("Model or class names not available. Please check paths.")
        else:
            with st.spinner("Classifying …"):
                img_tensor = preprocess_image(pil_img, (img_size_opt, img_size_opt))
                probs = model.predict(img_tensor, verbose=0)[0]

            pred_idx   = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx]) * 100

            # ── Prediction box ────────────────────────────────────────────────
            st.markdown(
                f'<div class="pred-box">'
                f'<h3>🐠 {pred_class}</h3>'
                f'<p>Predicted species</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="conf-box">'
                f'<h3>{confidence:.1f}%</h3>'
                f'<p>Model confidence</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Confidence gauge
            st.progress(int(confidence))

            # ── Top-K chart ───────────────────────────────────────────────────
            st.subheader(f"📊 Top-{top_k} Confidence Scores")
            fig = make_confidence_bar(class_names, probs, top_k)
            st.plotly_chart(fig, use_container_width=True)

            # ── Full probability table ────────────────────────────────────────
            with st.expander("📋 All class probabilities"):
                import pandas as pd
                df = pd.DataFrame({
                    "Species": class_names,
                    "Confidence (%)": (probs * 100).round(2),
                }).sort_values("Confidence (%)", ascending=False).reset_index(drop=True)
                st.dataframe(df, use_container_width=True)

# ── Sample gallery hint ───────────────────────────────────────────────────────
else:
    st.markdown("### 👆 Upload an image to get started")
    st.markdown(
        """
        This application uses a deep learning model trained on multiple fish species.
        Simply upload an image and the model will:
        - 🔍 Identify the fish species
        - 📊 Show confidence scores for all species
        - 📋 Provide a full probability breakdown
        """
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Multiclass Fish Image Classification | "
    "Built with TensorFlow/Keras + Streamlit | "
    "Transfer Learning: VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0"
)
