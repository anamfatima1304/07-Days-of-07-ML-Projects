## Iris Classification Model

**Model Overview**

This repository hosts a Python-based Iris classification model. The project explores three machine learning algorithms: Decision Tree, K-Nearest Neighbors (KNN), and Logistic Regression. After comparative analysis, Logistic Regression was selected as the optimal model for classifying Iris species. The model is deployed using Streamlit for a user-friendly graphical interface.

**Dataset**

The model utilizes the renowned Iris dataset, a classic benchmark in machine learning. It contains measurements of sepal length, sepal width, petal length, and petal width for 150 Iris flowers, categorized into three species: Iris setosa, Iris versicolor, and Iris virginica.

**Model Development**

1. **Data Preprocessing**: The dataset is loaded and preprocessed, handling potential missing values and outliers.
2. **Model Training**: Three models (Decision Tree, KNN, Logistic Regression) are trained and evaluated using appropriate metrics (accuracy, precision, recall, F1-score).
3. **Model Selection**: Logistic Regression demonstrates superior performance and is chosen as the final model.
4. **Model Deployment**: The selected model is integrated into a Streamlit application, providing a user-friendly interface for Iris species prediction.

**Usage**

To run the Streamlit app:

1. Clone this repository.
2. Install required libraries: `pip install streamlit pandas numpy scikit-learn`
3. Navigate to the project directory.
4. Run `streamlit run iris_classifier.py`

The Streamlit app allows users to input Iris flower measurements and receive predicted species.

