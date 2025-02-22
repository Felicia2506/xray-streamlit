#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os

# Google Drive File ID for the new model
GDRIVE_FILE_ID = "1-BCKd-ssavT3O8HQ-NSeP1fuswuMDeMa"
MODEL_FILE = "xray_classifier.tflite"

def download_model():
    """Download the TFLite model from Google Drive if not present."""
    if not os.path.exists(MODEL_FILE):
        st.write("Downloading model...")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        response = requests.get(url, stream=True)
        with open(MODEL_FILE, "wb") as f:
            f.write(response.content)
        st.write("Download complete!")
    return MODEL_FILE

@st.cache_resource
def load_model():
    """Load the TFLite model (download if necessary)."""
    model_path = download_model()
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, input_shape):
    """
    Resize and normalize the image.
    Assumes input_shape is in the format [1, height, width, channels]
    """
    image = image.resize((input_shape[1], input_shape[2]))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(interpreter, processed_image):
    """Run inference and return prediction probabilities."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# Define class labels for the Xray Classifier
CLASS_LABELS = {0: "safe", 1: "gun", 2: "knife"}

st.title("Xray Classifier")
st.write("Upload an X-ray image to view prediction probabilities.")

# Load model
interpreter = load_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image based on the model's input shape
    input_details = interpreter.get_input_details()
    processed_image = preprocess_image(image, input_details[0]['shape'])
    
    # Get prediction probabilities
    predictions = predict_image(interpreter, processed_image)
    
    st.write("### Prediction Probabilities")
    for idx, prob in enumerate(predictions):
        st.write(f"{CLASS_LABELS.get(idx, 'Unknown')}: {prob:.4f}")

