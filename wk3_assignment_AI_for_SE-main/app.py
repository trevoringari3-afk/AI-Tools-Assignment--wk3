import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title
st.title(" MNIST Digit Classifier")
st.write("Upload a handwritten digit image (0–9) and the AI will guess what it is.")

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn.h5")

# Upload image
uploaded_file = st.file_uploader("Upload a digit image (PNG or JPG)", type=["png", "jpg", "jpeg"])

def preprocess_image(img):
    # Convert to grayscale
    img = img.convert("L")
    # Resize to 28x28 pixels
    img = img.resize((28, 28))
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Reshape to (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_digit = np.argmax(prediction)

    # Show result
    st.markdown(f"###  Predicted Digit: **{predicted_digit}**")
