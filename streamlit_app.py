import streamlit as st
from PIL import Image

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
