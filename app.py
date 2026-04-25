import streamlit as st
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import soundfile as sf
import tempfile
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
SAMPLE_RATE = 16000
MODEL_PATHS = {
    "RandomForest": "rf_model.pkl",
    "SVM": "svm_model (1).pkl",
    "LogisticRegression": "lr_model.pkl",
}

CLASS_NAMES = {
    0: "Tamil",
    1: "Telugu",
    2: "Kannada",
    3: "Hindi",
    4: "Malayalam",
}

# Load models
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = joblib.load(path)
            st.success(f"Loaded {name}")
        except Exception as e:
            st.error(f"Could not load {name}: {e}")
    return models

models = load_models()

# Feature extraction functions
def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160, win_length=400)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = {}
    for i in range(n_mfcc):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
        features[f'mfcc_d_{i+1}_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_d_{i+1}_std'] = np.std(mfcc_delta[i])
        features[f'mfcc_d2_{i+1}_mean'] = np.mean(mfcc_delta2[i])
        features[f'mfcc_d2_{i+1}_std'] = np.std(mfcc_delta2[i])
    return features

def extract_pitch(y, sr):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, y, sr)

    try:
        sound = parselmouth.Sound(tmp_path)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch.selected_array['frequency']
        voiced = pitch_values[pitch_values > 0]

        if len(voiced) == 0:
            features = {
                'pitch_mean': 0.0, 'pitch_std': 0.0,
                'pitch_min': 0.0, 'pitch_max': 0.0,
                'voiced_fraction': 0.0
            }
        else:
            features = {
                'pitch_mean': float(np.mean(voiced)),
                'pitch_std': float(np.std(voiced)),
                'pitch_min': float(np.min(voiced)),
                'pitch_max': float(np.max(voiced)),
                'voiced_fraction': float(len(voiced) / len(pitch_values))
            }
    except:
        features = {
            'pitch_mean': 0.0, 'pitch_std': 0.0,
            'pitch_min': 0.0, 'pitch_max': 0.0,
            'voiced_fraction': 0.0
        }
    finally:
        os.remove(tmp_path)
    return features

def extract_energy(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    features = {
        'energy_mean': float(np.mean(rms)),
        'energy_std': float(np.std(rms)),
        'energy_max': float(np.max(rms)),
        'energy_dynamic_range': float(np.max(rms) - np.min(rms))
    }
    return features

def extract_formants(y, sr):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, y, sr)

    try:
        sound = parselmouth.Sound(tmp_path)
        formants = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        n_frames = call(formants, "Get number of frames")

        f1_vals, f2_vals, f3_vals = [], [], []
        for frame in range(1, n_frames + 1):
            f1 = call(formants, "Get value at time", 1, call(formants, "Get time from frame number", frame), 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, call(formants, "Get time from frame number", frame), 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, call(formants, "Get time from frame number", frame), 'Hertz', 'Linear')

            if f1 and not np.isnan(f1): f1_vals.append(f1)
            if f2 and not np.isnan(f2): f2_vals.append(f2)
            if f3 and not np.isnan(f3): f3_vals.append(f3)

        features = {
            'F1_mean': float(np.mean(f1_vals)) if f1_vals else 0.0,
            'F1_std': float(np.std(f1_vals)) if f1_vals else 0.0,
            'F2_mean': float(np.mean(f2_vals)) if f2_vals else 0.0,
            'F2_std': float(np.std(f2_vals)) if f2_vals else 0.0,
            'F3_mean': float(np.mean(f3_vals)) if f3_vals else 0.0,
            'F3_std': float(np.std(f3_vals)) if f3_vals else 0.0,
        }
    except:
        features = {
            'F1_mean': 0.0, 'F1_std': 0.0,
            'F2_mean': 0.0, 'F2_std': 0.0,
            'F3_mean': 0.0, 'F3_std': 0.0,
        }
    finally:
        os.remove(tmp_path)
    return features

def extract_spectral(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    features = {
        'spectral_centroid_mean': float(np.mean(centroid)),
        'spectral_centroid_std': float(np.std(centroid)),
        'spectral_bandwidth_mean': float(np.mean(bandwidth)),
        'spectral_bandwidth_std': float(np.std(bandwidth)),
        'spectral_rolloff_mean': float(np.mean(rolloff)),
        'spectral_rolloff_std': float(np.std(rolloff)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
    }
    return features

def extract_phonotactic(y, sr):
    frame_len = 400
    hop_len = 160
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]

    rms_norm = rms / (np.max(rms) + 1e-8)
    zcr_norm = zcr / (np.max(zcr) + 1e-8)

    vowel_frames = np.sum((rms_norm > 0.3) & (zcr_norm < 0.3))
    consonant_frames = np.sum((rms_norm < 0.3) | (zcr_norm > 0.5))
    total_frames = len(rms)

    vc_ratio = vowel_frames / (consonant_frames + 1e-8)

    features = {
        'vowel_ratio': float(vowel_frames / total_frames),
        'consonant_ratio': float(consonant_frames / total_frames),
        'vc_ratio': float(vc_ratio),
    }
    return features

def extract_features(y, sr):
    features = {}
    features.update(extract_mfcc(y, sr))
    features.update(extract_energy(y, sr))
    features.update(extract_spectral(y, sr))
    features.update(extract_phonotactic(y, sr))
    features.update(extract_pitch(y, sr))
    features.update(extract_formants(y, sr))
    return features

# Streamlit app
st.title("Language Classification App")
st.write("Upload a .wav audio file to predict the language using trained ML models.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE, mono=True)
    st.audio(uploaded_file, format='audio/wav')

    # Extract features
    with st.spinner("Extracting features..."):
        feat_dict = extract_features(y, sr)
        X = np.array(list(feat_dict.values())).reshape(1, -1)

    st.write("Features extracted successfully!")

    # Predictions
    st.subheader("Predictions")
    for model_name, pipeline in models.items():
        try:
            pred = pipeline.predict(X)[0]
            prob = pipeline.predict_proba(X)[0] if hasattr(pipeline, 'predict_proba') else None
            lang = CLASS_NAMES.get(pred, f"Class {pred}")
            st.write(f"**{model_name}**: {lang}")
            if prob is not None:
                st.write(f"Probabilities: {dict(zip([CLASS_NAMES[i] for i in range(len(prob))], prob))}")
        except Exception as e:
            st.error(f"Error with {model_name}: {e}")

st.write("Note: This app uses models trained on selected features. Predictions may not be accurate if the feature set differs.")