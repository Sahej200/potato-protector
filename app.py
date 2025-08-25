import os
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image

# Lazy import tensorflow so the app can still render the UI if TF isn't installed yet
try:
    import tensorflow as tf  # type: ignore
except Exception as e:
    tf = None

st.set_page_config(page_title="Potato Protector – Potato Leaf Disease", page_icon="🥔")

st.title("🥔 Crop Protector")
st.caption("Predict potato leaf disease using a pre-trained Keras/TensorFlow model.")

# Resolve model path relative to this file
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "potatoes.h5"

model_path = st.sidebar.text_input("Model file (.h5)", str(DEFAULT_MODEL_PATH), help="Path to your Keras model. Defaults to potatoes.h5 next to app.py")
compile_model = st.sidebar.checkbox("Compile model on load (usually not needed)", value=False, help="If your model needs custom losses/metrics you can toggle this on.")

# Try loading model
model = None
model_load_error = None
input_hw = (224, 224)  # sensible default
n_classes = None

if tf is None:
    model_load_error = "TensorFlow is not installed. Open a terminal and run:  pip install tensorflow"
else:
    try:
        model = tf.keras.models.load_model(model_path, compile=compile_model)
        # Try to infer input size
        try:
            ishape = getattr(model, "input_shape", None)
            if isinstance(ishape, (list, tuple)):
                # handle models with multiple inputs by picking the first
                if isinstance(ishape[0], (list, tuple)):
                    ishape = ishape[0]
            # ishape like (None, H, W, C) for channels-last
            if ishape and len(ishape) >= 4:
                H, W = ishape[1], ishape[2]
                if isinstance(H, int) and isinstance(W, int):
                    input_hw = (H, W)
        except Exception:
            pass
        # Infer number of classes from last layer output shape
        try:
            oshape = getattr(model, "output_shape", None)
            if isinstance(oshape, (list, tuple)):
                if isinstance(oshape[0], (list, tuple)):
                    oshape = oshape[0]
            if oshape and len(oshape) >= 2 and isinstance(oshape[-1], int):
                n_classes = oshape[-1]
        except Exception:
            pass
    except Exception as e:
        model_load_error = f"Failed to load model from '{model_path}'. Error: {e}"

with st.expander("ℹ️ Model details", expanded=False):
    st.write(f"**Model path:** `{model_path}`")
    st.write(f"**Expected input size (H×W):** {input_hw[0]}×{input_hw[1]} (inferred or default)")
    st.write("**Output classes:** " + (str(n_classes) if n_classes is not None else "Unknown"))
    if model_load_error:
        st.error(model_load_error)
    else:
        st.success("Model loaded successfully.")

# Class labels configuration
default_labels = None
if n_classes is not None and n_classes == 3:
    # Common labels for Potato Leaf dataset – adjust if needed
    default_labels = ["Early Blight", "Late Blight", "Healthy"]
elif n_classes is not None and n_classes == 2:
    default_labels = ["Diseased", "Healthy"]
elif n_classes is not None and n_classes > 0:
    default_labels = [f"Class {i}" for i in range(n_classes)]

labels = []
if n_classes:
    st.sidebar.subheader("Class labels")
    for i in range(n_classes):
        default = default_labels[i] if (default_labels and i < len(default_labels)) else f"Class {i}"
        labels.append(st.sidebar.text_input(f"Label for class {i}", default))
else:
    st.sidebar.info("Load a model to configure class labels.")

def preprocess(img: Image.Image, size_hw=(224, 224)):
    # Ensure RGB, resize, scale to [0,1]
    img = img.convert("RGB")
    img = img.resize((size_hw[1], size_hw[0]))  # PIL uses (W,H)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def predict_image(img: Image.Image):
    if model is None or tf is None:
        raise RuntimeError("Model not loaded. Check TensorFlow installation and model path.")
    x = preprocess(img, input_hw)
    preds = model.predict(x)
    # Handle various output shapes
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0]
    elif preds.ndim == 1:
        probs = preds
    else:
        probs = preds.squeeze()
    # Binary case
    if probs.ndim == 0:
        probs = np.array([1.0 - float(probs), float(probs)])
    if probs.size == 1:
        probs = np.array([1.0 - float(probs[0]), float(probs[0])])
    # Softmax if not already
    if probs.ndim == 1:
        # ensure non-negative and normalize
        probs = np.array(probs, dtype="float32")
        if np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0, atol=1e-3):
            ex = np.exp(probs - np.max(probs))
            probs = ex / np.sum(ex)
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    return top_idx, top_prob, probs

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict"):
        try:
            idx, conf, probs = predict_image(image)
            if labels and idx < len(labels):
                pred_name = labels[idx]
            else:
                pred_name = f"Class {idx}"
            st.subheader(f"Prediction: {pred_name}")
            st.write(f"Confidence: {conf*100:.2f}%")
            # Show probabilities table
            if n_classes:
                st.write("Class probabilities:")
                for i, p in enumerate(probs):
                    name = labels[i] if (labels and i < len(labels)) else f"Class {i}"
                    st.write(f"- {name}: {float(p)*100:.2f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload an image to get started.")
