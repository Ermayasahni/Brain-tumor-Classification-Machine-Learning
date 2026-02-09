import streamlit as st
import joblib
import os
import numpy as np
from PIL import Image

model = joblib.load("Brain_tumor.joblib")


def predict(image):
    
    img = image.convert("L")
    img_resized = img.resize((200, 200))
    img_array = np.array(img_resized)
    img_flat = img_array.flatten() / 255.0  
    img_resh = img_flat.reshape(1, -1)

     
    output_array = model.predict(img_resh)
    output = output_array[0]

    
    if output == 0:
        return "No Tumor"
    elif output == 1:
        return "Positive Tumor"
    else:
        return "Unknown"



st.title("Brain Tumor Classifier")
st.markdown("Upload an MRI image to classify it as 'No Tumor' or 'Positive Tumor'.")


uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image")
    
    
    if st.button("Check Result"):
        result = predict(image)
        st.markdown(f"Prediction: {result}")
    
