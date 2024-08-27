import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import exposure

# Load your pre-trained model
model = tf.keras.models.load_model('nn_model.h5')

# Define class names
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

def preprocess_image(image):
    img_gray = image.convert('L')  # 'L' mode is for grayscale
    img_resized = img_gray.resize((70, 70))
    img_array = img_to_array(img_resized)
    img_array_normalized = img_array.astype('float32') / 255.0
    img_eq = exposure.equalize_hist(img_array_normalized)
    return img_eq

def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict the class
    img = np.expand_dims(processed_image, axis=0) 
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Return the class name
    return class_names[predicted_class[0]]

# Streamlit App
st.title("Image Classification App")
st.write("Upload an X-Ray image to classify it as COVID, Normal, Pneumonia, or Lung Opacity.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Predict the class
    st.write("Classifying...")
    label = predict(image)
    
    # Show the result
    st.write(f"Prediction: **{label}**")
