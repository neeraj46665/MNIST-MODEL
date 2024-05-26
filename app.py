import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
# (Assuming the model is saved as 'mnist_model.h5' after training)
model = load_model('mnist_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a numpy array and scale the pixel values to [0, 1]
    image = np.array(image).astype('float32') / 255
    # Reshape the array to (1, 28, 28, 1)
    image = np.reshape(image, (1, 28, 28, 1))
    return image

# Streamlit Sidebar
st.sidebar.title("About MNIST Dataset")
st.sidebar.info(
    """
    The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. 
    It was created by Yann LeCun, Corinna Cortes, and Christopher Burges. 
    The dataset contains 60,000 training images and 10,000 testing images of digits from 0 to 9.
    """
)

st.sidebar.title("How to Use")
st.sidebar.info(
    """
    1. Upload an image of a handwritten digit.
    2. The image will be processed and displayed on the screen.
    3. The model will predict the digit in the uploaded image.
    """
)

# Streamlit Main Interface
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit, and the model will predict the digit.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    
    # Display the prediction
    st.write(f"Predicted Digit: **{predicted_digit}**")
