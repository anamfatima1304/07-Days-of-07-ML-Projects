# Image Classification App

This is a application that allows users to classify X-Ray images into one of four categories: **COVID**, **Lung Opacity**, **Normal**, or **Viral Pneumonia**. The application trains a neural network model to perform the classification.

## Features

- Upload an X-Ray image (JPEG or PNG).
- Display the uploaded image.
- Classify the image into one of the four categories.
- Display the predicted category.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/image-classification-app.git
   ```

3. To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```

Once the application is running, open your web browser and go to http://localhost:8501/. You can then upload an X-Ray image and see the classification result.

## Code Overview

- **Classifier.py**: The main file containing the Streamlit application code.
- **nn_model.h5**: The pre-trained neural network model used for classification.
- **MedicalImageClassification.ipynb**: The jupyter notebook for preprocessing and model building.
- **COVID_19_Radiography_Dataset** : The Dataset that ontains the image on which the model is built.

## Preprocessing and Prediction
The uploaded image undergoes the following preprocessing steps before being passed to the model:

- **Grayscale Conversion**: The image is converted to grayscale to match the input format expected by the model.
- **Resizing**: The image is resized to 70x70 pixels.
- **Normalization**: The pixel values are normalized to the range [0, 1].
- **Histogram Equalization**: The image histogram is equalized to improve contrast.

The processed image is then fed into the model to predict the class.

## Model
The neural network model used in this application is a Convolutional Neural Network (CNN) trained to classify X-Ray images into four categories: COVID, Lung Opacity, Normal, and Viral Pneumonia.


## Contributing

Contributions are always welcome!

Find out an Error or give any suggestion to improve the project.

If you want to add any new features, make a pull request.

ThankYou for your attention.

