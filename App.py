import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


# 1. Load the model (cached so it doesn't reload on every click)
@st.cache_resource
def load_my_model():
    # Make sure 'face_detection.h5' is in the same folder
    return tf.keras.models.load_model("face_detection.h5")


model = load_my_model()

# 2. App Title and Description
st.title("Face Mask Detection System 😷")
st.write("Upload an image to check if the person is wearing a mask.")

# 3. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 4. Image Preprocessing (Matches your Colab logic)
    st.write("Classifying...")

    # Convert PIL image to NumPy array
    img_array = np.array(image)

    # Handle if image is grayscale or RGBA
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Resize to 128x128 (This matches your notebook's input_image_resized)
    img_resized = cv2.resize(img_array, (128, 128))

    # Normalize pixel values (Matches input_image_scaled)
    img_scaled = img_resized / 255.0

    # Reshape for the model (1, 128, 128, 3)
    img_reshaped = np.reshape(img_scaled, [1, 128, 128, 3])

    # 5. Prediction
    prediction = model.predict(img_reshaped)
    pred_label = np.argmax(prediction)

    # 6. Display Result (0=No Mask, 1=Mask based on your labels)
    if pred_label == 1:
        st.success("Prediction: **Wearing a Mask** ")
    else:
        st.error("Prediction: **Not Wearing a Mask** ")
