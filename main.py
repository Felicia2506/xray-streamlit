#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import tensorflow.lite as tflite
import requests
from PIL import Image
import os

# Google Drive File ID for the new X-ray Classifier model
GDRIVE_FILE_ID = "1-BCKd-ssavT3O8HQ-NSeP1fuswuMDeMa"

@st.cache_resource
def download_tflite_model():
    """Downloads the TFLite model from Google Drive if not already present."""
    model_path = "xray_model.tflite"
    if not os.path.exists(model_path):
        st.write("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
        st.write("✅ Model downloaded successfully!")
    else:
        st.write("✅ Model already exists. Loading...")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load the model
interpreter = download_tflite_model()

# Define class labels for the X-ray Classifier
CLASS_LABELS = {
    0: "safe",
    1: "gun",
    2: "knife"
}

def preprocess_image(image, input_shape):
    """Resize, normalize, and add a batch dimension to the image."""
    image = image.resize((input_shape[1], input_shape[2]))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(interpreter, processed_image):
    """Run inference on the preprocessed image and return probability scores."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# Streamlit App UI
st.title("X-ray Classifier")
st.write("Upload an X-ray image to see the prediction probabilities.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image using model input shape
    input_details = interpreter.get_input_details()
    processed_image = preprocess_image(image, input_details[0]['shape'])
    
    # Get predictions
    predictions = predict(interpreter, processed_image)
    
    st.subheader("Prediction Probabilities")
    for idx, prob in enumerate(predictions):
        st.write(f"{CLASS_LABELS.get(idx, 'Unknown')}: {prob:.4f}")

