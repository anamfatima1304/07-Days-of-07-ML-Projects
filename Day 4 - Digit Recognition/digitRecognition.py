import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from PIL import Image
import io
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.h5')

class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  

def preprocess_image(img):
    # Convert image to gray_scale
    img = img.convert('L')

    # Convert to numpy array
    img_array = img_to_array(img)

    # Resize the image (MNIST images are 28x28)
    img_array = cv2.resize(img_array, (28, 28))

    # Normalize the image (values between 0 and 1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 28, 28)
    # Check and correct any possible issues (e.g., invalid pixel values)
    img_array = np.clip(img_array, 0, 1)  # Ensure pixel values are between 0 and 1
    # Apply Gaussian blur (optional)
    img_array = cv2.GaussianBlur(img_array[0], (3, 3), 0)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension back
    # Final check and ensure image shape matches model input
    if img_array.shape != (1, 28, 28):
        raise ValueError(f"Unexpected image shape: {img_array.shape}. Expected (1, 28, 28).")

    return img_array

def predict(img):
    """
    Predicts the class of the given image using the pre-trained model.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(img)
    # Make prediction
    predictions = model.predict(preprocessed_image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # Convert index to class label
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, predictions[0]

# Streamlit app
st.title('Handwritten Digit Recognition')

st.write("""
    **Upload an image of a handwritten digit (0-9) below, and I will classify it for you!**
""")

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict
    st.write("Classifying the uploaded image...")
    result, prediction_probs = predict(image)
    
    # Display the result
    st.markdown(f"### The predicted class is: **{result}**")
    
