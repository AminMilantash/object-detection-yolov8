import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("YOLOv8 Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = YOLO('models/best.pt')
    results = model.predict(image)

    st.image(results[0].plot(), caption="Detected Objects")
