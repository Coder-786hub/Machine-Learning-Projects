import os
import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load the model
model = joblib.load("model.joblib")

# Prediction function
def predict(image):
    # Convert the image to grayscale
    img = image.convert("L")
    img_resized = img.resize((200, 200))
    img_array = np.array(img_resized)
    img_flat = img_array.flatten() / 255.0  # Normalize the image data
    img_flat = img_flat.reshape(1, -1)
    
    # Get model prediction
    output_array = model.predict(img_flat)
    output = output_array[0]

    # Set result label based on prediction
    if output == 0:
        return "No Tumor"
    elif output == 1:
        return "Positive Tumor"
    else:
        return "Unknown"

# Streamlit GUI
st.title("Brain Tumor Classifier")
st.markdown("Upload an MRI image to classify it as 'No Tumor' or 'Positive Tumor'.")

# Uploading an MRI image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    # Predict button
    if st.button("Check Result"):
        result = predict(image)
        st.markdown(f"### Prediction: {result}")
