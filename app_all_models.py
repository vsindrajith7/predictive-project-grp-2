import streamlit as st
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import soundfile as sf
import tempfile
import os
import io
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
SAMPLE_RATE = 16000

CLASS_NAMES = {
    0: "Tamil",
    1: "Telugu",
    2: "Kannada",
    3: "Hindi",
    4: "Malayalam",
}

MODEL_PATHS = {
    "Random Forest": "rf_model.pkl",
    "SVM": "svm_model.pkl",
    "Logistic Regression": "lr_model.pkl"
}

def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

# Load all models
models = {}
for model_name, path in MODEL_PATHS.items():
    models[model_name] = load_model(path)

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

    # Add duration feature
    features['duration'] = len(y) / sr

    # Select only the 61 features that the models were trained on
    selected_features = [
        'duration', 'spectral_bandwidth_mean', 'F3_mean', 'spectral_centroid_mean',
        'spectral_centroid_std', 'spectral_rolloff_std', 'mfcc_2_mean', 'F2_mean',
        'mfcc_d2_1_std', 'spectral_bandwidth_std', 'mfcc_d_1_std', 'zcr_std',
        'mfcc_d2_5_std', 'zcr_mean', 'mfcc_d2_7_std', 'F1_std', 'mfcc_d2_2_std',
        'mfcc_1_std', 'mfcc_d2_3_std', 'mfcc_d_5_std', 'mfcc_3_mean', 'mfcc_d_7_std',
        'mfcc_d_11_std', 'mfcc_d2_8_std', 'mfcc_d2_11_std', 'mfcc_d2_9_std',
        'mfcc_d_8_std', 'mfcc_d_9_std', 'mfcc_d_2_std', 'mfcc_d2_13_std',
        'mfcc_d_13_std', 'mfcc_7_mean', 'F2_std', 'mfcc_6_mean', 'pitch_mean',
        'mfcc_11_mean', 'mfcc_2_std', 'mfcc_d_12_std', 'mfcc_d2_12_std', 'mfcc_9_std',
        'mfcc_7_std', 'energy_max', 'mfcc_d2_10_std', 'mfcc_5_std', 'energy_std',
        'mfcc_11_std', 'mfcc_10_mean', 'mfcc_d_10_std', 'mfcc_13_std', 'mfcc_12_std',
        'mfcc_9_mean', 'mfcc_d_3_std', 'mfcc_8_std', 'mfcc_d2_6_std', 'mfcc_13_mean',
        'F1_mean', 'energy_mean', 'vowel_ratio', 'mfcc_5_mean', 'mfcc_d2_4_std', 'F3_std'
    ]

    # Extract only the selected features in the correct order
    X = np.array([features[feat] for feat in selected_features])
    return X

# Streamlit app
st.set_page_config(
    page_title="AI Language Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for digital/tech theme
st.markdown("""
<style>
    /* Digital/tech theme */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Title styling */
    .title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #00d4ff, #090979);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #090979);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
        background: linear-gradient(45deg, #090979, #00d4ff);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(0, 212, 255, 0.1);
        border: 2px dashed rgba(0, 212, 255, 0.5);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(45deg, #00ff88, #009944);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #00d4ff, #090979);
    }
    
    /* Table styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(15, 15, 35, 0.9);
        border-right: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Model selection styling */
    .model-selector {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Animation for results */
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.5); }
        50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.8); }
        100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.5); }
    }
    
    .result-card {
        animation: glow 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with model info
with st.sidebar:
    st.markdown("## 🤖 AI Models")
    st.markdown("---")
    
    model_info = {
        "Random Forest": {"accuracy": "89.0%", "description": "Ensemble learning method"},
        "SVM": {"accuracy": "96.2%", "description": "Support Vector Machine"},
        "Logistic Regression": {"accuracy": "87.3%", "description": "Linear classification"}
    }
    
    for model_name, info in model_info.items():
        if models.get(model_name):
            st.markdown(f"**{model_name}**")
            st.markdown(f"📊 Accuracy: {info['accuracy']}")
            st.markdown(f"📝 {info['description']}")
            st.markdown("---")
        else:
            st.markdown(f"❌ {model_name} - Not Available")

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<h1 class="title">🎯 AI Language Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Machine Learning for Indian Language Recognition</p>', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📁 Upload Audio File")
uploaded_file = st.file_uploader(
    "Select a .wav audio file for language detection",
    type="wav",
    help="Upload speech audio in WAV format for accurate language recognition"
)

if uploaded_file is not None:
    st.markdown("✅ File uploaded successfully!")
    audio_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    # Audio player
    st.audio(audio_bytes, format='audio/wav')
    
    # Load audio
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    
    # Audio info
    duration = len(y) / sr
    st.info(f"🎵 Audio Duration: {duration:.2f} seconds | Sample Rate: {sr} Hz")

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Processing section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚡ Feature Extraction")
    
    with st.spinner("🔍 Analyzing audio features..."):
        progress_bar = st.progress(0)
        for i in range(100):
            # Simulate processing time
            import time
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        X = extract_features(y, sr).reshape(1, -1)
    
    st.markdown('<div class="success-message">✅ Feature extraction completed!</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model selection and prediction
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Language Detection")
    
    available_models = [name for name, model in models.items() if model is not None]
    
    if available_models:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Select AI Model")
            selected_model = st.selectbox(
                "",
                available_models,
                help="Choose the machine learning model for prediction"
            )
        
        with col2:
            st.markdown("#### Model Performance")
            if selected_model == "SVM":
                st.metric("Accuracy", "96.2%", "Best Model")
            elif selected_model == "Random Forest":
                st.metric("Accuracy", "89.0%", "High Performance")
            elif selected_model == "Logistic Regression":
                st.metric("Accuracy", "87.3%", "Good Performance")

        # Prediction button
        if st.button("🚀 Detect Language", key="predict"):
            with st.spinner("🤖 AI is analyzing..."):
                model = models[selected_model]
                prediction = model.predict(X)[0]
                language = CLASS_NAMES.get(int(prediction), f"Class {prediction}")
                
                # Show result with animation
                st.markdown(f'''
                <div class="result-card" style="background: linear-gradient(45deg, #00d4ff, #090979); 
                     color: white; padding: 2rem; border-radius: 15px; text-align: center; 
                     margin: 2rem 0; font-size: 2rem; font-weight: bold;">
                🎯 Detected Language: {language}
                </div>
                ''', unsafe_allow_html=True)
                
                # Show probabilities if available
                if hasattr(model, "predict_proba"):
                    st.markdown("#### 📊 Prediction Confidence")
                    probs = model.predict_proba(X)[0]
                    
                    # Create a nice probability display
                    prob_data = []
                    for i, prob in enumerate(probs):
                        lang_name = CLASS_NAMES.get(i, f"Class {i}")
                        prob_data.append({"Language": lang_name, "Confidence": f"{prob*100:.1f}%"})
                    
                    prob_df = pd.DataFrame(prob_data).sort_values("Confidence", ascending=False)
                    
                    # Display as progress bars
                    for _, row in prob_df.iterrows():
                        confidence = float(row["Confidence"].replace("%", ""))
                        st.markdown(f"**{row['Language']}**")
                        st.progress(confidence / 100)
                        st.markdown(f"{row['Confidence']}")
                        st.markdown("---")
    else:
        st.error("❌ No AI models are available. Please check model files.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Comparison section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔄 Model Comparison")
    
    if st.button("⚖️ Compare All Models", key="compare"):
        st.markdown("#### 🤖 AI Model Predictions")
        
        results = []
        for model_name, model in models.items():
            if model is not None:
                prediction = model.predict(X)[0]
                language = CLASS_NAMES.get(int(prediction), f"Class {prediction}")
                results.append({"🤖 Model": model_name, "🎯 Prediction": language})
            else:
                results.append({"🤖 Model": model_name, "🎯 Prediction": "❌ Not Available"})

        results_df = pd.DataFrame(results)
        
        # Style the table
        st.table(results_df.style.set_properties(**{
            'background-color': 'rgba(255, 255, 255, 0.05)',
            'color': 'white',
            'border-color': 'rgba(0, 212, 255, 0.3)'
        }))
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔬 Powered by Advanced Machine Learning • Built with Streamlit</p>
    <p>🎵 Supports Tamil, Telugu, Kannada, Hindi, Malayalam</p>
</div>
""", unsafe_allow_html=True)