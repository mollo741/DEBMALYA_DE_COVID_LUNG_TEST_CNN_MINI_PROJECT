import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="COVID-19 X-Ray Detection",
    page_icon="🩻",
    layout="centered"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🩻 COVID-19 Chest X-Ray Detection")

st.write(
    "Upload a chest X-ray image and the trained ResNet50 "
    "model will classify it as COVID-19, Normal, or "
    "Viral Pneumonia."
)


# ==========================================================
# MODEL SETTINGS
# ==========================================================

MODEL_PATH = "covid_resnet50.keras"

GOOGLE_DRIVE_FILE_ID = (
    "1IeU2iiTMURWaoZJ_LwhqjvU27za8u9Q-"
)


# ==========================================================
# DOWNLOAD AND LOAD MODEL
# ==========================================================

@st.cache_resource
def download_and_load_model():

    # Download model if it doesn't already exist
    if not os.path.exists(MODEL_PATH):

        with st.spinner(
            "Downloading ResNet50 model..."
        ):

            gdown.download(
                id=GOOGLE_DRIVE_FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    # Load trained ResNet50 model
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    with st.spinner("Loading ResNet50 model..."):

        model = download_and_load_model()

    st.success(
        "ResNet50 model loaded successfully!"
    )

except Exception as e:

    st.error(
        "Unable to load the ResNet50 model."
    )

    st.write(
        "Please check that your Google Drive file "
        "is shared as 'Anyone with the link → Viewer'."
    )

    st.code(str(e))

    st.stop()


# ==========================================================
# CLASS NAMES
# ==========================================================

class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]


# ==========================================================
# IMAGE UPLOADER
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ------------------------------------------------------
    # Display uploaded image
    # ------------------------------------------------------

    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        width=450
    )


    # ------------------------------------------------------
    # Resize image to 224 × 224
    # ------------------------------------------------------

    image_resized = image.resize(
        (224, 224)
    )


    # ------------------------------------------------------
    # Convert image to NumPy
    # ------------------------------------------------------

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )


    # ------------------------------------------------------
    # Add batch dimension
    # ------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ------------------------------------------------------
    # ResNet50 preprocessing
    # ------------------------------------------------------

    image_array = preprocess_input(
        image_array
    )


    # ------------------------------------------------------
    # Make prediction
    # ------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )


    # ------------------------------------------------------
    # Get predicted class
    # ------------------------------------------------------

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = class_names[
        predicted_index
    ]


    # ------------------------------------------------------
    # Calculate confidence
    # ------------------------------------------------------

    confidence = (
        predictions[0][predicted_index] * 100
    )


    # ======================================================
    # DISPLAY PREDICTION
    # ======================================================

    st.subheader("Prediction")

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # ======================================================
    # CLASS PROBABILITIES
    # ======================================================

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(
        class_names
    ):

        probability = predictions[0][i]

        st.write(
            f"**{class_name}: "
            f"{probability * 100:.2f}%**"
        )

        st.progress(
            float(probability)
        )


# ==========================================================
# DISCLAIMER
# ==========================================================

st.markdown("---")

st.warning(
    "⚠️ This application is for educational purposes only "
    "and is not a medical diagnostic tool."
)

st.caption(
    "ResNet50 Transfer Learning | "
    "COVID-19 Chest X-Ray Classification"
)