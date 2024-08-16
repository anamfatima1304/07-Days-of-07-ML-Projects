# Movie Sentiment Analysis

## Overview

This project implements a sentiment analysis application for movie reviews. The application uses machine learning models to classify reviews as either positive or negative. The models used in this project include:

- **multimonial NB**
- **Logistic Regression**

After evaluating the performance of these models, Logistic Regression was selected as the best performing model. The application is built using Streamlit for an interactive web interface where users can input movie reviews and receive sentiment classifications.

## Features

- **Interactive User Interface:** Built using Streamlit, allowing users to enter movie reviews and get instant sentiment feedback.
- **Sentiment Classification:** The model classifies reviews as **Positive** or **Negative** based on the sentiment expressed.
- **Image Display:** Displays images based on the sentiment of the review, providing visual feedback.

## Usage
Run the Application:

Start the Streamlit application by executing:

```bash
streamlit run app.py
```
### Interact with the App:

Open the URL provided by Streamlit in your web browser.
Enter a movie review in the text area.
Click "Submit" to see the sentiment classification along with a relevant image.

## Model Evaluation
The models were evaluated based on their accuracy and performance metrics. Logistic Regression was selected for its balance of simplicity and performance.

## Contributing
If you have suggestions or improvements, feel free to fork the repository and submit a pull request.
