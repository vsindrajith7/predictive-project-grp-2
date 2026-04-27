# Model Retraining Guide: Adding English Language Support

## Current Status
✅ App is **configured** to recognize 6 languages (Tamil, Telugu, Kannada, Hindi, Malayalam, **English**)
⚠️ Model currently trained on 5 classes (needs retraining with English data)

## Steps to Retrain Model with English

### Prerequisites
Ensure you have these files/folders:
- Raw audio data in `data/raw/` directory for all 6 languages
- Feature extraction scripts working correctly
- All Python dependencies installed

### Step 1: Run Feature Extraction (One-time)
Run the feature engineering notebook to extract acoustic features from all audio files:

```bash
# Convert notebook to Python and execute
jupyter nbconvert --to notebook --execute feature_engineering.ipynb --inplace
```

This generates:
- `features_raw.csv` (all 104 features)
- `features_selected.csv` (61 selected features)

Both with 6-class labels: `['tamil', 'telugu', 'kannada', 'hindi', 'malayalam', 'english']`

### Step 2: Train Models with 6 Classes
Once feature CSVs are created, retrain the ML models:

```bash
jupyter nbconvert --to notebook --execute "ML Pipeline.ipynb" --inplace
```

This creates new model files:
- `rf_model.pkl` - Random Forest (6-class)
- `lr_model.pkl` - Logistic Regression (6-class)
- `cv_results.csv` - Updated cross-validation results

### Step 3: Verify the App Works
Upload an English audio file (.wav or .mp3) to test the app. It should now correctly detect English!

## What Happens During Retraining

1. **Feature Extraction** (~30-60 min depending on data size)
   - Loads all audio files from `data/raw/{language}/`
   - Extracts 104 acoustic features per file
   - Outputs CSV with features for 6 languages

2. **Model Training** (~10-30 min)
   - Loads feature CSVs
   - Trains 3 classifiers: Random Forest, SVM, Logistic Regression
   - Performs 5-fold cross-validation
   - Tests on held-out test set
   - Saves best models as `.pkl` files

## Troubleshooting

### "File not found" errors
- Check that your current working directory is the project root
- Ensure `data/raw/` contains language subdirectories with audio files
- Verify CSV paths in notebooks match your setup

### Feature extraction is slow
- Normal for thousands of audio files (~5-10 min per 1000 files)
- Pitch and formant extraction (via Praat) are the slowest steps
- CPU usage will be high; this is expected

### Model training is slow
- Grid search hyperparameter tuning can take time
- SVM is slowest on large datasets (~10-20 min)
- Random Forest is typically faster (~5-10 min)

## Quick Start Script

Use the provided `retrain_model_with_english.py` script:

```bash
python retrain_model_with_english.py
```

This automatically:
1. Extracts features from all 6 languages
2. Trains models on the 6-class dataset
3. Replaces old `.pkl` files with new 6-class versions

## Verification

After retraining, verify the model has 6 classes:

```python
import joblib
model = joblib.load('rf_model.pkl')
print(model.steps[1][1].n_classes_)  # Should print: 6
print(model.steps[1][1].classes_)     # Should print: [0 1 2 3 4 5]
```

## Rolling Back

If you need to revert to the 5-class model:
- Keep backups of old `.pkl` files before retraining
- Or restore from git: `git checkout rf_model.pkl lr_model.pkl`
