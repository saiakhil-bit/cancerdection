import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO('best_model.pt')

st.title("YOLO Object Detection")
st.write("Upload an image or use a webcam to detect objects using YOLO.")

# Choose input method
input_option = st.selectbox("Choose input", ("Image Upload", "Webcam"))

# Detect objects from uploaded image
if input_option == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert image to OpenCV format
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Run detection
        results = model(image_cv2)
        
        # Display results
        st.image(results[0].plot(), caption="Detection Results", use_column_width=True)

# Detect objects from webcam
elif input_option == "Webcam":
    cap = cv2.VideoCapture(0)

    if st.button("Start Webcam"):
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to retrieve frame. Please refresh the page.")
                break

            # Run YOLO detection
            results = model(frame)
            detected_frame = results[0].plot()

            # Show the frame
            stframe.image(detected_frame, channels="BGR")

    cap.release()
