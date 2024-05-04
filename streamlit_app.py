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



import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

# Load the model
model = torch.hub.load('ultralytics/yolov8', 'yolov8_nano')
model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS
model.load_state_dict(torch.load('best_yolov8nano.pt'))
model.eval()

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Convert the image to a tensor
    img_tensor = F.to_tensor(image)
    
    # Perform inference
    results = model(img_tensor, size=640)  # includes NMS

    # Process detections
    for label, score, (x1, y1, x2, y2) in results.xyxy[0]:
        if label == 0:  # assuming 'person' class is label 0
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
