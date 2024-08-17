import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('DecisionTree.pkl')  
vectorizer = joblib.load('Scaler.pkl')
pca = joblib.load('PCA.pkl') 

# Title of the app
st.title('Wine Quality Prediction')

# Instructions for the user
st.markdown("""
    **Welcome to the Wine Quality Prediction App!**
    
    Please enter the following features of the wine to get a prediction of its quality.
    """)

# Input fields for user to enter the features
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, format="%.2f")
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, format="%.2f")
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, format="%.2f")
chlorides = st.number_input("Chlorides", min_value=0.0, format="%.2f")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, format="%.2f")
density = st.number_input("Density", min_value=0.0, format="%.4f")
pH = st.number_input("pH", min_value=0.0, format="%.2f")
alcohol = st.number_input("Alcohol", min_value=0.0, format="%.2f")

# Button to predict
if st.button("Predict Wine Quality"):
    # Prepare the input for prediction
    features = np.array([[
        fixed_acidity, volatile_acidity, residual_sugar, chlorides,
        total_sulfur_dioxide, density, pH, alcohol
    ]])
    
    # Predict the quality of the wine
    scaler = vectorizer.transform(features)
    PCA = pca.transform(scaler)
    prediction = model.predict(PCA)
    
    # Display the result
    st.write(f'### Predicted Wine Quality: {prediction[0]}')
    
    # Display an image based on the quality (optional)
    if prediction[0] <= 5:
        st.image('low_quality_image.jfif', caption='Low Quality Wine', use_column_width=True)
    else:
        st.image('high_quality_image.jfif', caption='High Quality Wine', use_column_width=True)
