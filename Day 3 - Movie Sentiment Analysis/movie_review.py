import streamlit as st
import pickle
from PIL import Image
import numpy as np

import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import re
import base64


# Downloading necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

model = joblib.load('best_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Title of the app
st.title('Movie Review Sentiment Analysis')

# Instructions for the user
st.markdown("""
    **Welcome to the Sentiment Analysis App!**
    
    Please enter a movie review below, and I will classify it as **Positive** or **Negative**.
    """, unsafe_allow_html=True)

# User input
review = st.text_area("Enter your movie review:")

# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 
    
# Function to preprocess data
def preprocess(text):
  # Lowering the data
  lowered_text = text.lower()
  # Removing Punctuation
  punctuation_removed = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), lowered_text))
  # Tokenization
  tokenized_text = word_tokenize(punctuation_removed)
  # Stop Words removal
  stop_words = set(stopwords.words('english'))
  Removed_text = [w for w in tokenized_text if not w in stop_words]
  # POS Tagging
  Tagged_Words = pos_tag(Removed_text)
  # Lemmatization
  lemmatized_tokens = []
  lemmatizer = WordNetLemmatizer()
  for token, tag in Tagged_Words:
    wordnet_pos = get_wordnet_pos(tag)
    # Lemmatize the word with the correct POS tag
    lemmatized_word = lemmatizer.lemmatize(token, wordnet_pos)
    lemmatized_tokens.append(lemmatized_word)
  # Spelling Correction
  corrected_spellings = [Word(token).correct() for token in lemmatized_tokens]
  return corrected_spellings


# Function to predict sentiment
def predict_sentiment(text):
    text = preprocess(text)
    text = vectorizer.transform(text)
    prediction = model.predict(text)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Predict sentiment if review is provided
if review:
    sentiment = predict_sentiment(review)
    
    # Show result
    st.write(f'### The review is classified as: **{sentiment}**')
    
    # Display a relevant image based on sentiment
    if sentiment == 'Positive':
        st.image('positive_image.jpg', caption='Happy Movie!', use_column_width=True)
    else:
        st.image('negative_image.jpg', caption='Sad Movie!', use_column_width=True)
else:
    st.write("Please enter a review to get started!")
