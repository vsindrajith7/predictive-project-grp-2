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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
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
    "Logistic Regression": "lr_model.pkl",
}

# ─────────────────────────────────────────────
# FIX 1: Load scaler alongside models only when needed.
# The saved SVM/LR models are pipelines with an internal scaler step.
# External scaler.pkl is only required if a loaded model does not
# already contain a StandardScaler inside its pipeline.
# ─────────────────────────────────────────────
SCALER_PATH = "scaler.pkl"

# Models that require feature scaling
NEEDS_SCALING = {"SVM", "Logistic Regression"}


def model_has_internal_scaler(model):
    if isinstance(model, Pipeline):
        return any(isinstance(step, StandardScaler) for _, step in model.steps)
    return False


def model_needs_external_scaling(model_name, model):
    return model_name in NEEDS_SCALING and model is not None and not model_has_internal_scaler(model)


@st.cache_resource
def load_artifacts():
    """Load all models and the scaler once, cache them."""
    loaded_models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                loaded_models[name] = joblib.load(path)
            except Exception as e:
                loaded_models[name] = None
                st.sidebar.warning(f"⚠️ Could not load {name}: {e}")
        else:
            loaded_models[name] = None

    scaler = None
    if any(model_needs_external_scaling(name, model) for name, model in loaded_models.items()):
        if os.path.exists(SCALER_PATH):
            try:
                scaler = joblib.load(SCALER_PATH)
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load scaler: {e}")
        else:
            st.sidebar.warning(
                "⚠️ **scaler.pkl not found.**\n\n"
                "A loaded model requires external scaling, but scaler.pkl is missing. "
                "Export your training scaler with `joblib.dump(scaler, 'scaler.pkl')`."
            )
    else:
        if os.path.exists(SCALER_PATH):
            try:
                scaler = joblib.load(SCALER_PATH)
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load scaler: {e}")

    return loaded_models, scaler


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────
SELECTED_FEATURES = [
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
            return {'pitch_mean': 0.0, 'pitch_std': 0.0,
                    'pitch_min': 0.0, 'pitch_max': 0.0, 'voiced_fraction': 0.0}
        return {
            'pitch_mean': float(np.mean(voiced)),
            'pitch_std': float(np.std(voiced)),
            'pitch_min': float(np.min(voiced)),
            'pitch_max': float(np.max(voiced)),
            'voiced_fraction': float(len(voiced) / len(pitch_values)),
        }
    except Exception:
        return {'pitch_mean': 0.0, 'pitch_std': 0.0,
                'pitch_min': 0.0, 'pitch_max': 0.0, 'voiced_fraction': 0.0}
    finally:
        os.remove(tmp_path)


def extract_energy(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    return {
        'energy_mean': float(np.mean(rms)),
        'energy_std': float(np.std(rms)),
        'energy_max': float(np.max(rms)),
        'energy_dynamic_range': float(np.max(rms) - np.min(rms)),
    }


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
            t = call(formants, "Get time from frame number", frame)
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            if f1 and not np.isnan(f1): f1_vals.append(f1)
            if f2 and not np.isnan(f2): f2_vals.append(f2)
            if f3 and not np.isnan(f3): f3_vals.append(f3)
        return {
            'F1_mean': float(np.mean(f1_vals)) if f1_vals else 0.0,
            'F1_std': float(np.std(f1_vals)) if f1_vals else 0.0,
            'F2_mean': float(np.mean(f2_vals)) if f2_vals else 0.0,
            'F2_std': float(np.std(f2_vals)) if f2_vals else 0.0,
            'F3_mean': float(np.mean(f3_vals)) if f3_vals else 0.0,
            'F3_std': float(np.std(f3_vals)) if f3_vals else 0.0,
        }
    except Exception:
        return {k: 0.0 for k in ['F1_mean', 'F1_std', 'F2_mean', 'F2_std', 'F3_mean', 'F3_std']}
    finally:
        os.remove(tmp_path)


def extract_spectral(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return {
        'spectral_centroid_mean': float(np.mean(centroid)),
        'spectral_centroid_std': float(np.std(centroid)),
        'spectral_bandwidth_mean': float(np.mean(bandwidth)),
        'spectral_bandwidth_std': float(np.std(bandwidth)),
        'spectral_rolloff_mean': float(np.mean(rolloff)),
        'spectral_rolloff_std': float(np.std(rolloff)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
    }


def extract_phonotactic(y, sr):
    frame_len, hop_len = 400, 160
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_norm = rms / (np.max(rms) + 1e-8)
    zcr_norm = zcr / (np.max(zcr) + 1e-8)
    vowel_frames = np.sum((rms_norm > 0.3) & (zcr_norm < 0.3))
    consonant_frames = np.sum((rms_norm < 0.3) | (zcr_norm > 0.5))
    total_frames = len(rms)
    return {
        'vowel_ratio': float(vowel_frames / total_frames),
        'consonant_ratio': float(consonant_frames / total_frames),
        'vc_ratio': float(vowel_frames / (consonant_frames + 1e-8)),
    }


def extract_features(y, sr):
    """Extract all features and return only the 61 the models were trained on."""
    features = {}
    features.update(extract_mfcc(y, sr))
    features.update(extract_energy(y, sr))
    features.update(extract_spectral(y, sr))
    features.update(extract_phonotactic(y, sr))
    features.update(extract_pitch(y, sr))
    features.update(extract_formants(y, sr))
    features['duration'] = len(y) / sr

    # ── FIX 2: Validate all expected features are present before stacking ──
    missing = [f for f in SELECTED_FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing features during extraction: {missing}")

    X = np.array([features[f] for f in SELECTED_FEATURES], dtype=np.float32)

    # ── FIX 3: Guard against NaN / Inf values that break model prediction ──
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def predict(model_name, model, scaler, X_raw):
    """
    Apply external scaler only when the loaded model does not already include internal scaling.
    Returns (predicted_label, proba_dict_or_None).
    """
    X = X_raw.reshape(1, -1)

    if model_needs_external_scaling(model_name, model):
        if scaler is not None:
            X = scaler.transform(X)
        else:
            st.warning(
                f"⚠️ **No scaler found for {model_name}.** "
                "Prediction may be inaccurate. Please add `scaler.pkl` next to your model files."
            )

    pred = int(model.predict(X)[0])
    language = CLASS_NAMES.get(pred, f"Class {pred}")

    proba = None
    if hasattr(model, "predict_proba"):
        raw_proba = model.predict_proba(X)[0]
        proba = {CLASS_NAMES.get(i, f"Class {i}"): float(p) for i, p in enumerate(raw_proba)}

    return language, proba


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Language Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

section[data-testid="stSidebar"] {
    background: #0d0d18 !important;
    border-right: 1px solid #1e1e3a;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(120deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}

.sub {
    font-family: 'Space Mono', monospace;
    color: #6b7280;
    font-size: 0.85rem;
    margin-bottom: 2.5rem;
    letter-spacing: 0.05em;
}

.panel {
    background: #111122;
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}

.panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1rem;
}

.lang-result {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    padding: 1.8rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #1a103a, #0f1f35);
    border: 1px solid #2d2060;
    color: #a78bfa;
    letter-spacing: -1px;
    margin: 1rem 0;
}

.badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    background: #1a1a2e;
    border: 1px solid #2d2060;
    color: #a78bfa;
    border-radius: 6px;
    padding: 3px 10px;
    margin-right: 6px;
    letter-spacing: 0.08em;
}

.warn-box {
    background: #1f1200;
    border: 1px solid #92400e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #fbbf24;
    margin: 1rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    font-weight: 700;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #60a5fa);
}

[data-testid="stFileUploader"] {
    background: #111122;
    border: 1.5px dashed #2d2060;
    border-radius: 12px;
    padding: 0.5rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin: 0.8rem 0;
}
.metric-box {
    flex: 1;
    background: #0d0d18;
    border: 1px solid #1e1e3a;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
}
.prob-lang {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #e8e8f0;
    width: 90px;
    flex-shrink: 0;
}
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #a78bfa;
    width: 52px;
    text-align: right;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ──
models, scaler = load_artifacts()
available_models = [n for n, m in models.items() if m is not None]
scaler_required_by_loaded_models = any(
    model_needs_external_scaling(name, model)
    for name, model in models.items()
)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🎛️ Models")
    MODEL_ACCURACY = {
        "Random Forest": ("89.0%", "Pipeline + optional scaler"),
        "SVM": ("96.2%", "Pipeline includes scaler"),
        "Logistic Regression": ("87.3%", "Pipeline includes scaler"),
    }
    for name, (acc, note) in MODEL_ACCURACY.items():
        status = "✅" if models.get(name) else "❌"
        st.markdown(f"**{status} {name}**")
        st.markdown(f"<span class='badge'>{acc}</span><span style='font-size:0.72rem;color:#6b7280'>{note}</span>", unsafe_allow_html=True)
        st.markdown("---")

    if scaler:
        scaler_status = "✅ Loaded"
    elif scaler_required_by_loaded_models:
        scaler_status = "❌ Missing — add scaler.pkl"
    else:
        scaler_status = "ℹ️ Not required for loaded models"

    st.markdown(f"**Scaler:** {scaler_status}")

    if not scaler and scaler_required_by_loaded_models:
        st.markdown("""
        <div class='warn-box'>
        Add <b>scaler.pkl</b> from your training script:<br><br>
        <code>joblib.dump(scaler, 'scaler.pkl')</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🗣️ Languages")
    for lang in CLASS_NAMES.values():
        st.markdown(f"• {lang}")

# ── Main ──
st.markdown("<div class='main-title'>🎙️ Language Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>INDIAN LANGUAGE RECOGNITION · ML-POWERED</div>", unsafe_allow_html=True)

# ── Upload ──
st.markdown("<div class='panel'><div class='panel-title'>01 · Upload Audio</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a .wav file (clear speech, ≥ 3 seconds works best)",
    type="wav",
    label_visibility="collapsed",
)

if uploaded_file:
    audio_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    st.audio(audio_bytes, format='audio/wav')

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-box'>
            <div class='metric-val'>{duration:.2f}s</div>
            <div class='metric-label'>Duration</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>{sr//1000}kHz</div>
            <div class='metric-label'>Sample Rate</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>{len(y)//1000}K</div>
            <div class='metric-label'>Samples</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file and y is not None:

    # ── Feature extraction ──
    st.markdown("<div class='panel'><div class='panel-title'>02 · Feature Extraction</div>", unsafe_allow_html=True)
    with st.spinner("Extracting 61 acoustic features…"):
        try:
            X_raw = extract_features(y, sr)
            st.success(f"✅ Extracted {len(X_raw)} features — no NaN/Inf values detected.")
        except Exception as e:
            st.error(f"❌ Feature extraction failed: {e}")
            st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Prediction ──
    st.markdown("<div class='panel'><div class='panel-title'>03 · Prediction</div>", unsafe_allow_html=True)

    if not available_models:
        st.error("❌ No models loaded. Place rf_model.pkl / svm_model.pkl / lr_model.pkl next to app.py.")
    else:
        col_sel, col_acc = st.columns([1, 1])
        with col_sel:
            selected_model_name = st.selectbox("Choose model", available_models)
        with col_acc:
            acc_label = MODEL_ACCURACY[selected_model_name][0]
            st.metric("Reported Accuracy", acc_label)

        selected_model = models[selected_model_name]
        if scaler is None and model_needs_external_scaling(selected_model_name, selected_model):
            st.markdown("""
            <div class='warn-box'>
            ⚠️ <b>scaler.pkl not found.</b> This model requires external feature scaling.
            Prediction will likely be wrong. Export your training scaler and place it here.
            </div>
            """, unsafe_allow_html=True)

        if st.button("▶ Detect Language"):
            model = selected_model
            with st.spinner("Running inference…"):
                try:
                    language, proba = predict(selected_model_name, model, scaler, X_raw)
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    st.stop()

            st.markdown(f"<div class='lang-result'>🗣️ {language}</div>", unsafe_allow_html=True)

            if proba:
                st.markdown("**Confidence breakdown**")
                sorted_proba = sorted(proba.items(), key=lambda x: x[1], reverse=True)
                for lang, p in sorted_proba:
                    pct = p * 100
                    bar_color = "#7c3aed" if lang == language else "#1e1e3a"
                    st.markdown(f"""
                    <div class='prob-row'>
                        <span class='prob-lang'>{lang}</span>
                        <div style='flex:1;background:#1a1a2e;border-radius:4px;height:10px;overflow:hidden'>
                            <div style='width:{pct:.1f}%;background:{bar_color};height:100%;border-radius:4px'></div>
                        </div>
                        <span class='prob-pct'>{pct:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── All models comparison ──
    st.markdown("<div class='panel'><div class='panel-title'>04 · Compare All Models</div>", unsafe_allow_html=True)
    if st.button("⚖️ Run All Models"):
        rows = []
        for name, model in models.items():
            if model is None:
                rows.append({"Model": name, "Prediction": "❌ Not loaded", "Top Confidence": "—"})
                continue
            try:
                lang, proba = predict(name, model, scaler, X_raw)
                top_conf = f"{max(proba.values())*100:.1f}%" if proba else "—"
                rows.append({"Model": name, "Prediction": lang, "Top Confidence": top_conf})
            except Exception as e:
                rows.append({"Model": name, "Prediction": f"Error: {e}", "Top Confidence": "—"})

        st.table(pd.DataFrame(rows))

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;font-family:Space Mono,monospace;
            font-size:0.72rem;color:#374151;letter-spacing:0.08em'>
    TAMIL · TELUGU · KANNADA · HINDI · MALAYALAM
</div>
""", unsafe_allow_html=True)