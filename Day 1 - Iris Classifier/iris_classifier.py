import streamlit as st
import joblib
import matplotlib.pyplot as plt
import io
from PIL import Image

# Load the models
Scaler_model = joblib.load('Scaling_model.pkl')  
Prediction_model = joblib.load('Prediction_model.pkl')  

# Define the flower names
Numbers_to_class = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

def plot_image(flower_class):
    # Create an image based on the predicted flower class
    fig, ax = plt.subplots()
    img = Image.open(f'{flower_class}.jfif')  
    # ax.imshow(img)
    # ax.axis('off')
    # return fig
    return img

# Streamlit app
st.title('Iris Flower Classifier')

# User inputs
Sepal_Length = st.number_input("Enter the Length of Sepal in cm:", format="%.2f")
Sepal_Width = st.number_input("Enter the Width of Sepal in cm:", format="%.2f")
Petal_Length = st.number_input("Enter the Length of Petal in cm:", format="%.2f")

if st.button('Predict'):
    scale = Scaler_model.transform([[Sepal_Length, Sepal_Width, Petal_Length]])
    prediction = Prediction_model.predict(scale)
    flower_class = Numbers_to_class[prediction[0]]
    
    # Display the prediction
    st.write(f"The Flower is {flower_class}")
    
    # Plotting the Image
    img = plot_image(flower_class)
    st.image(img, caption=flower_class, width = 400)
