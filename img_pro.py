import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Function to perform image processing
def process_image(image, techniques):
    if 'Resize' in techniques:
        width, height = st.slider("Select resize dimensions", 1, 1000, (image.width, image.height))
        image = image.resize((width, height))

    if 'Grayscale Conversion' in techniques:
        image = image.convert('L')

    if 'Image Cropping' in techniques:
        left, top, right, bottom = st.slider("Select crop region", 0, image.width, (0, image.width))
        image = image.crop((left, top, right, bottom))

    if 'Image Rotation' in techniques:
        angle = st.slider("Select rotation angle", -180, 180, 0)
        image = image.rotate(angle)

    return image

# Main Streamlit app
st.title("Image Processing App")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display original image
if uploaded_image is not None:
    original_image = Image.open(uploaded_image)
    st.subheader("Original Image")
    st.image(original_image, caption="Original Image", use_column_width=True)

    # Select image processing techniques
    selected_techniques = st.multiselect("Select image processing techniques", 
                                         ['Resize', 'Grayscale Conversion', 'Image Cropping', 'Image Rotation'])

    # Process image
    if st.button("Process Image"):
        processed_image = process_image(original_image, selected_techniques)
        st.subheader("Processed Image")
        st.image(processed_image, caption="Processed Image", use_column_width=True)
