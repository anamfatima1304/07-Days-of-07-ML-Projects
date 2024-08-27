import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import io
import soundfile as sf
import sounddevice as sd
import pandas as pd

# Load the pre-trained model and scaler
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('StandardScaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

genre_mapping = ['blues', 'classical', 'country','disco','hiphop','jazz','metal','pop','reggae','rock']

# Function to Pre Process the Data
def PreProcessing(audio_file):
    audio, sr = librosa.load(audio_file, sr = 22059)
    audio_trimmed, _ = librosa.effects.trim(audio)
    desired_length = 30 * sr  # For 30 seconds
    if len(audio) > desired_length:
        audio = audio_trimmed[:desired_length]  # Truncate
    else:
        audio = np.pad(audio_trimmed, (0, max(0, desired_length - len(audio))), mode='constant')  # Pad
    return audio, sr

# Function to extract features
def extract_features(y, sr):
    # 1. Chroma Features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    chroma_stft_mean = np.mean(chroma_stft_mean)
    chroma_stft_var = np.var(chroma_stft, axis=1)
    chroma_stft_var = np.mean(chroma_stft_var)
    
    # 2. RMS (Root Mean Square) Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    
    # 3. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)
    
    # 4. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    
    # 5. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    
    # 6. Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)
    
    # 7. Harmony
    harmony = librosa.effects.harmonic(y=y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    
    # 8. Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # 9. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)

    # 10. Perceptual Spread 
    perceptual_spread = librosa.feature.spectral_contrast(y=y, sr=sr)
    perceptr_mean = np.mean(perceptual_spread)
    perceptr_var = np.var(perceptual_spread)
    
    # Creating features dictionary
    features = {
        'length': len(y),
        'chroma_stft_mean': chroma_stft_mean,
        'chroma_stft_var': chroma_stft_var,
        'rms_mean': rms_mean,
        'rms_var': rms_var,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_centroid_var': spectral_centroid_var,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'spectral_bandwidth_var': spectral_bandwidth_var,
        'rolloff_mean': rolloff_mean,
        'rolloff_var': rolloff_var,
        'zero_crossing_rate_mean': zero_crossing_rate_mean,
        'zero_crossing_rate_var': zero_crossing_rate_var,
        'harmony_mean': harmony_mean,
        'harmony_var': harmony_var,
        'tempo': tempo,
        'perceptr_mean': perceptr_mean, 
        'perceptr_var': perceptr_var,  
        **{f'mfcc{i+1}_mean': mfcc_means[i] for i in range(20)},
        **{f'mfcc{i+1}_var': mfcc_vars[i] for i in range(20)},
    }
    
    return features

# Function to predict Genre 
def predict(audio_path):
    audio, sr = PreProcessing(audio_path)
    features = extract_features(audio, sr)
    features_DataFrame = pd.DataFrame(features)

    new_order = ['length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean',
        'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
        'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var' ,'perceptr_mean', 'perceptr_var', 'tempo',
        'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
        'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
        'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
        'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
        'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
        'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
        'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
        'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var' ]

    df_reordered = features_DataFrame.reindex(columns=new_order)

    scaled_features = loaded_scaler.transform(df_reordered)

    prediction = loaded_model.predict(scaled_features) # True Prediction
    return prediction , audio, sr

# Streamlit App
st.title('Music Genre Classification App')

# Upload audio file
uploaded_file = st.file_uploader("Upload a music file", type=["wav", "mp3", "ogg"])

# If a file is uploaded
if uploaded_file is not None:
    prediction, y, sr = predict(uploaded_file)
    genre = prediction[0]
    
    # Display the predicted genre
    st.write(f"**Predicted Genre:** {genre_mapping[genre]}")

    # Display waveform
    st.subheader("Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    plt.title('Waveform')
    st.pyplot(fig)

    # Option to play/stop audio
    st.subheader("Play/Stop Audio")
    play_button = st.button("Play")
    stop_button = st.button("Stop")

    # Audio playback
    if play_button:
        sd.play(y, sr)
    if stop_button:
        sd.stop()

else:
    st.write("Please upload a music file to analyze.")

