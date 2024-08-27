# Music Classifier

## Overview

The Music Classifier project aims to classify music into one of ten predefined categories using machine learning techniques. The project utilizes audio features extracted with the `librosa` library and provides a user-friendly interface through a Streamlit application.

The categories for music classification are:

- **Blues**
- **Classical**
- **Country**
- **Disco**
- **Hip-hop**
- **Jazz**
- **Metal**
- **Pop**
- **Reggae**
- **Rock**

## Features

- **Feature Extraction:** Extract meaningful audio features from music files using `librosa`.
- **Model Training:** Train a machine learning model to classify music into the ten categories.
- **User Interface:** A Streamlit application that allows users to upload music files and view the classification results.

## Dataset

The dataset used for training and evaluating the classifier consists of audio files labeled with one of the ten categories. Each audio file is categorized and used to train the model to recognize different music genres. The audio features were also already available in **features_30_sec** file.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/anamfatima1304/30-Days-of-30-Projects.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd Day 6 - Music Classifier
   ```

3. **Install Dependencies:**

   Create a virtual environment (optional) and install the required packages:

   ```bash
   pip install numpy pandas scikit-learn librosa streamlit matplotlib
   ```

## Usage

1. **Run the Jupyter Notebook:**

   Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open the `MusicClassifier.ipynb` notebook. This notebook contains the code for preprocessing the audio data, feature extraction, and model training. Follow the steps in the notebook to preprocess the data and train the model.

2. **Run the Streamlit Application:**

   To launch the Streamlit application, run the following command:

   ```bash
   streamlit run Classifier_Interface.py
   ```

   This command will start a local server, and you can access the Streamlit app by opening your web browser and navigating to `http://localhost:8801`.

3. **Using the Streamlit App:**

   - **Upload Music File:** Click the file upload button to select a music file from your local machine.
   - **View Classification:** The app will process the uploaded music file, classify it into one of the ten categories, and display the result.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Make sure your code adheres to the existing style and includes appropriate documentation and tests.

## Acknowledgements

- [Librosa Documentation](https://librosa.org/doc/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
