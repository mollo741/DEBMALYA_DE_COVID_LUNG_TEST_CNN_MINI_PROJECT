import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="COVID-19 X-Ray Detection",
    page_icon="🩻",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🩻 COVID-19 Chest X-Ray Detection")

st.write(
    "Upload a chest X-ray image and the ResNet50 model "
    "will classify it as COVID-19, Normal, or Viral Pneumonia."
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "covid_resnet50.keras"
    )


model = load_model()


# --------------------------------------------------
# Class Names
# IMPORTANT: Must match training order
# --------------------------------------------------

class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="Chest X-Ray",
        width=400
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert to NumPy
    image_array = np.array(image_resized)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ResNet50 preprocessing
    image_array = preprocess_input(
        image_array.astype(np.float32)
    )

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = (
        predictions[0][predicted_index] * 100
    )


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.subheader("Prediction")

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # --------------------------------------------------
    # Display All Probabilities
    # --------------------------------------------------

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):

        probability = predictions[0][i]

        st.write(
            f"**{class_name}:** "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.warning(
    "⚠️ This application is for educational purposes "
    "only and is not a medical diagnostic tool."
)