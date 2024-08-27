import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Values of all The Categories to create Drop Downs
Brand_Array = [
    'Alfa', 'Aston', 'Audi', 'BMW', 'Bentley', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ferrari',
    'Ford', 'GMC', 'Genesis', 'Honda', 'Hummer', 'Hyundai', 'INFINITI', 'Jaguar', 'Jeep', 'Karma', 'Kia', 'Lamborghini',
    'Land', 'Lexus', 'Lincoln', 'Lotus', 'Lucid', 'MINI', 'Maserati', 'Maybach', 'Mazda', 'McLaren', 'Mercedes-Benz',
    'Mercury', 'Mitsubishi', 'Nissan', 'Plymouth', 'Polestar', 'Pontiac', 'Porsche', 'RAM', 'Rivian', 'Rolls-Royce',
    'Saab', 'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo', 'smart']
Fuel_Array  = ['E85 Flex Fuel', 'Gasoline', 'Hybrid', 'Plug-In Hybrid']
Transmission_Array = ['1-Speed Automatic', '10-Speed A/T', '10-Speed Automatic', '4-Speed A/T', '5-Speed A/T', '5-Speed M/T',
    '6-Speed A/T', '6-Speed Automatic', '6-Speed M/T', '7-Speed A/T', '7-Speed Automatic with Auto-Shift',
    '8-Speed A/T', '8-Speed Automatic', '9-Speed A/T', '9-Speed Automatic', 'A/T', 'Automatic', 'Automatic CVT',
    'CVT Transmission', 'M/T', 'Others', 'Transmission w/Dual Shift Mode']
EnginType_Array = ['Flat 6','I3','I4','I6','None','Turbo','V-10','V-12','V-6','V-8','V10','V12','V6','V8']

with open('Model_values.txt', 'r') as file:
    content = file.read()
values = content.split(',')
Model_Array = [value.strip() for value in values] 

# Load the trained Gradient Boosting Regressor model
with open('GradientBoost.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the Trained Scaler
with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to get user input from the app interface
def user_input_features():
    model_year = st.slider('Model Year', 1990, 2024, 2020)
    milage = st.number_input('Mileage (miles)', 0, 300000, 15000)
    horsepower = st.number_input('Horsepower', 50, 1000, 200)
    displacement_liters = st.number_input('Displacement (Liters)', 0.5, 10.0, 2.0)
    cylinders = st.number_input('Number of Cylinders', 1, 10, 1)
    # Brand
    brand = st.selectbox('Brand', (b for b in Brand_Array)) 
    brand_type_encoded = [1 if b == brand else 0 for b in Brand_Array]

    # Fuel Type
    fuel_type = st.selectbox('Fuel Type', (Fuel for Fuel in Fuel_Array))
    fuel_type_encoded = [1 if fuel == fuel_type else 0 for fuel in Fuel_Array]

    # Model
    model_type = st.selectbox('Model', (m for m in Model_Array))
    model_encoded = [1 if model == model_type else 0 for model in Model_Array]

    # Transmission
    transmission = st.selectbox('Transmission', (t for t in Transmission_Array))
    transmission_encoded = [1 if trans == transmission else 0 for trans in Transmission_Array]

    # Accident
    accident = st.selectbox('Accident', (
        'None reported', 'At least 1 accident or damage reported'))
    accident_encoded = 1 if accident == 'None reported' else 0

    # Engine Type
    Engine_Type = st.selectbox('Engine Type', (e for e in EnginType_Array))
    engine_encoded = [1 if eng == Engine_Type else 0 for eng in EnginType_Array]


    # Construct the DataFrame from user inputs
    data = {
        'model_year': model_year,
        'milage': milage,
        'horsepower': horsepower,
        'displacement_liters': displacement_liters,
        'cylinders' : cylinders,
        **dict(zip(Brand_Array, brand_type_encoded)),

        **dict(zip(Fuel_Array, fuel_type_encoded)),

        **dict(zip(Model_Array, model_encoded)),

        **dict(zip(Transmission_Array, transmission_encoded)),

    'None reported' : accident_encoded,
    **dict(zip(EnginType_Array , engine_encoded)),

    }

    return pd.DataFrame(data, index=[0])

st.title("Car Price Prediction")

# Get user input
input_df = user_input_features()

# Predict using the model
if st.button('Predict'):
    # Scaling the Required Columns
    columns = ['model_year', 'milage', 'horsepower', 'displacement_liters','cylinders']
    array = []

    for column in columns:
        array.append(float(input_df[column]))

    scaled_array = scaler.transform([array])

    # Adding the Scaled values to original Dataframe
    new_values = scaled_array[0]
    for column in columns:
        input_df[column] = new_values[0]

    # Predicting the Values
    prediction = model.predict(input_df.values)
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")
