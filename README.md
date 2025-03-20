# 🎤 Parkinson's Detection from Voice using speech patterns

This project uses **CNN + XGBoost** to detect Parkinson's disease from voice recordings. The system extracts various audio features and predicts whether the individual is likely to have Parkinson's disease.

## 📚 Dataset Used
The model was trained using the **Parkinson's Dataset** which contains voice measurements from healthy and Parkinson’s patients. The data includes features extracted from sustained phonation of the vowel `/a/`.

- **Dataset Path:** `parkinsons.data`
- **Target Variable:** `status` (0 = Healthy, 1 = Parkinson’s)
- **Feature Count:** 22 audio features

---

## 🚀 Key Features
- 💄 **Upload Audio File** – Upload a `.wav` audio file.
- 🧐 **Real-time Prediction** – Analyze the uploaded file and predict Parkinson’s status.
- 📈 **Audio Analysis Visualization** – Display waveform and spectrogram for audio analysis.
- 🎯 **Model Accuracy** – Achieved 94% accuracy.

---

## 🛠️ Models and Methods Used
### 1. **Convolutional Neural Network (CNN)**
- Extracts deep audio features.
- Architecture:
  - 3 convolutional layers with ReLU activation.
  - Max-pooling applied after each convolution.
  - Dense feature extraction layer (`feature_layer`).
  - Final dense layer with sigmoid activation for binary classification.

### 2. **XGBoost Classifier**
- Trained on deep features extracted from the CNN.
- Used for the final classification.

### 🧩 **Feature Extraction Methods:**
- **Pitch Features:** YIN algorithm for pitch extraction.
- **MFCC (Mel-frequency cepstral coefficients):** Extracts 13 MFCCs to capture audio characteristics.
- **Short-term Energy Entropy:** Derived from pyAudioAnalysis.
- **Harmonic-to-Noise Ratio (HNR):** Measures the ratio of harmonic to percussive components.
- **Tempo Feature:** Estimated using beat tracking.

---

## 🎯 Model Performance
- ✅ **CNN + XGBoost Accuracy:** 94%
- 🔍 **Evaluation Metrics:**
  - Precision, Recall, and F1-score evaluated using classification reports.

---

## 💂️ Project Structure
```
/parkinsons-voice-detection
├── /models
│   ├── scaler.pkl              # Scaler for feature normalization
│   ├── feature_extractor.pkl   # CNN model for deep feature extraction
│   └── xgb_model.pkl           # XGBoost classifier model
├── /data
│   └── parkinsons.data         # Dataset used for training
├── app.py                      # Streamlit app for prediction
└── README.md                   # Project documentation
```

---






## ▶️ Run the Application
```bash
streamlit run app.py
```


## 📊 Model Training
### CNN Model Training:
- Data normalized using `StandardScaler`.
- CNN architecture built with TensorFlow and Keras.
- Deep features extracted from the intermediate dense layer (`feature_layer`).
- Trained with 30 epochs and batch size of 8.

### XGBoost Training:
- XGBoost classifier trained on extracted deep features.
- Evaluation performed using classification metrics and cross-validation.

---

## 📝 Usage Instructions
1. Launch the Streamlit application.
2. Upload an audio file in `.wav` format.
3. The model extracts audio features and displays:
    - Prediction result with confidence score.
    - Audio waveform and spectrogram analysis.


---
![Screenshot 2025-03-20 155743](https://github.com/user-attachments/assets/fcb5374a-2f93-488c-9ec5-e197014235d2)
![image](https://github.com/user-attachments/assets/93d7dcb3-0775-47e6-9ff8-5c3922127d34)
