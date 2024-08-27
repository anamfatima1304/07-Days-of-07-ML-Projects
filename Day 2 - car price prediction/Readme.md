# Car Price Prediction

## Overview

The Car Price Prediction project aims to predict the price of used cars based on various features using machine learning techniques. This project utilizes a dataset from Kaggle and applies different regression models to forecast car prices.

## Features

- **Data Preprocessing:** Clean and preprocess the dataset, including handling missing values and feature engineering.
- **Model Training:** Train multiple regression models to predict car prices.
- **Evaluation:** Evaluate model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
- **Prediction:** Make predictions based on user input or updated data.

## Dataset

The project uses the "Used Car Price Prediction" dataset from Kaggle, which includes various features related to used cars. Features in the dataset include:

- Car brand
- Model
- Year of manufacture
- Mileage
- Engine size
- Fuel type
- Transmission type
- Number of doors
- Price

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/anamfatima1304/30-Days-of-30-Projects.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd Day 2 - car price prediction
   ```

3. **Install Dependencies:**

   Create a virtual environment (optional) and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can manually install the dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

## Usage

1. **Prepare the Dataset:**

   Place your dataset file (e.g., `car_prices.csv`) in the project directory.

2. **Run the Jupyter Notebook:**

   Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open the `CarPricePrediction.ipynb` notebook and execute the cells to preprocess the data, train the models, and make predictions.

3. **Prediction Example:**

   If you have an updated dataset or new data, you can update the DataFrame with new values and use the trained model to make predictions. For example:

   ```python
   import pandas as pd
   from sklearn.ensemble import RandomForestRegressor

   # Load the dataset
   df = pd.read_csv('car_prices.csv')

   # Update specific columns
   df.loc[0, ['Mileage', 'Engine_Size']] = [50000, 2.0]

   # Load the trained model
   model = RandomForestRegressor()
   model.load('model.pkl')  # Replace with your model file path

   # Make predictions
   predictions = model.predict(df[['Mileage', 'Engine_Size']])
   print(predictions)
   ```


## Contributing

If you want to contribute to this project, please fork the repository and submit a pull request with your changes. Ensure that your code follows the existing style and includes appropriate tests.


## Acknowledgements

- [Kaggle Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
