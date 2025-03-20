import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import joblib
import os

# Load models and scaler
scaler = joblib.load("scaler.pkl")
feature_extractor = joblib.load("feature_extractor.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Streamlit UI Title
st.title("🎤 Parkinson's Detection from Voice")
st.write("Upload an audio file to analyze for Parkinson’s symptoms using advanced machine learning models.")

# --- Low-pass filter for noise reduction ---
def apply_lowpass_filter(y, sr, cutoff=500):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(6, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, y)

# --- Feature Extraction ---
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=44100)
        y = apply_lowpass_filter(y, sr, cutoff=500)
        features = []

        # Pitch-based Features
        yin_values = librosa.yin(y, fmin=190, fmax=210, sr=sr)
        features.extend([np.mean(yin_values), np.max(yin_values), np.min(yin_values)])
        features.append(np.std(librosa.feature.zero_crossing_rate(y=y)))
        features.append(np.std(librosa.feature.rms(y=y)))

        # MFCC Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1).tolist())

        # Short-term Features from pyAudioAnalysis
        [fs, x] = audioBasicIO.read_audio_file(file_path)
        x = audioBasicIO.stereo_to_mono(x)
        short_features, _ = ShortTermFeatures.feature_extraction(x, fs, 0.050 * fs, 0.025 * fs)

        energy_entropy = np.mean(short_features[1]) if len(short_features) > 1 else 0
        features.append(energy_entropy)

        # Pitch & HNR Features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if pitches.any() else 0
        features.append(pitch_mean)

        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.sum(np.abs(harmonic)) / (np.sum(np.abs(percussive)) + 1e-6)
        features.append(hnr)

        # Tempo Feature
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo))

        # Return features if extraction was successful
        return np.array(features) if len(features) == 22 else None
    except Exception as e:
        st.error(f"⚠️ Feature extraction error: {e}")
        return None

# --- Prediction Logic ---
def predict_parkinsons(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    deep_features = feature_extractor.predict(features_scaled.reshape(1, -1, 1))
    prediction_proba = xgb_model.predict_proba(deep_features)
    prediction = xgb_model.predict(deep_features)[0]

    confidence = max(prediction_proba[0]) * 100
    result = "🟢 Healthy" if prediction == 0 else "🔴 Parkinson’s Detected"
    return result, confidence

# --- Audio EDA Visualization ---
def plot_audio_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    plt.title("📈 Waveform of Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

def plot_spectrogram(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.title("🎧 Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    st.pyplot(plt)

# --- File Upload and Processing ---
uploaded_file = st.file_uploader("📤 Upload an audio file (WAV format)", type=["wav"])

if uploaded_file:
    temp_file = "temp_audio.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract Audio Features
    extracted_features = extract_audio_features(temp_file)
    if extracted_features is not None:
        # Prediction
        result, confidence = predict_parkinsons(extracted_features)
        st.subheader("🧠 Prediction Result")
        st.success(f"{result} with {confidence:.2f}% confidence")

        # EDA Visualization
        st.subheader("📊 Audio Analysis")
        y, sr = librosa.load(temp_file, sr=44100)
        plot_audio_waveform(y, sr)
        plot_spectrogram(y, sr)
    else:
        st.error("❌ Unable to extract audio features. Please try another file.")
