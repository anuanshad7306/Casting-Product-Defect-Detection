import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
MODEL_PATH = r"C:\\Users\\Pc\\Downloads\\Casting_Project\\casting_data\\casting_defect_model.h5"
model = load_model(MODEL_PATH)

# Define class names
class_names = ["Defective", "Non Defective"]

st.title("Casting Product Defect Detection")
st.write("Upload an image to classify it as Defective or Non Defective.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)
    
    # Convert image to RGB if it has an alpha channel (RGBA)
    if image_display.mode != 'RGB':
        image_display = image_display.convert('RGB')
    
    # Preprocess the image
    img = image_display.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)[0][0]

    # Display prediction result
    result = class_names[int(prediction > 0.5)]
    st.write(f"### Prediction: {result}")
