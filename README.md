Streamlit app link: https://predictive-project-grp-2-4qim8cuvksfzxdc7hsmj9h.streamlit.app/
### DESCRIPTION OF MEMBER1- TASK (DATASET COLLECTION, EDA&PREPROCESSING, FEATURE_ENGINEERING)
## DATASET COLLECTION (dataset.py)
## OBJECTIVE
Collect, extract, convert, filter, and organize raw audio clips into a clean, structured dataset ready for preprocessing and feature extraction in subsequent stages.
## DATA SOURCE
<img width="747" height="211" alt="image" src="https://github.com/user-attachments/assets/26496fee-69ca-4b00-acbf-e47ae24956f8" />

---

## Steps Performed in This Notebook

### Cell 1 — Install dependencies
Installs `librosa`, `soundfile`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `requests`.

### Cell 2 — Create folder structure
Creates all required directories under `data/raw/`, `data/processed/`, `data/metadata/`, `data/splits/`, and `downloads/`.

### Cell 3 — Define OpenSLR download sources
Maps each language to its verified OpenSLR download URL using the EU mirror (`openslr.elda.org`) for reliability.

### Cell 4 — Download all datasets
Downloads `.zip` and `.tar.gz` archives for all 5 languages with a progress bar. Skips files already downloaded.

### Cell 5 — Extract archives
Extracts all downloaded archives into the corresponding `data/raw/{language}/` folders.

### Cell 6 — Flatten nested folder structure
OpenSLR archives contain nested subfolders. This cell walks all subdirectories and copies audio files directly into `data/raw/{language}/`, capping at 300 clips per language.

### Cell 7 — Convert to 16 kHz mono WAV
Converts all `.mp3` and `.flac` files to 16 kHz mono WAV format using `librosa.load()` and `soundfile.write()`. This standardizes sample rate and channel count across all languages.

### Cell 8 — Filter by duration (3–10 seconds)
Removes clips shorter than 3 seconds or longer than 10 seconds. These are outside the target range for feature extraction.

### Cell 9 — Build metadata.csv
Walks all processed folders and builds a single CSV with the following columns:

| Column         | Description                          |
|----------------|--------------------------------------|
| `file_path`    | Relative path to the WAV file        |
| `file_name`    | Filename                             |
| `language`     | Language label                       |
| `duration_sec` | Duration of the clip in seconds      |
| `sample_rate`  | Always 16000 Hz                      |
| `source`       | Always `openslr`                     |

### Cell 10 — Sanity check
Plots one waveform per language to visually confirm audio loaded correctly.

---

## Output

| File | Description |
|------|-------------|
| `data/metadata/metadata.csv` | Master metadata file — primary handoff to next stage |
| `data/processed/{language}/*.wav` | Cleaned, standardized audio clips |

---

## Dataset Statistics (approximate)

| Language  | Clips | Avg Duration | Total Audio |
|-----------|-------|--------------|-------------|
| Malayalam | ~290  | ~5.2s        | ~25 min     |
| Tamil     | ~290  | ~5.1s        | ~25 min     |
| Hindi     | ~290  | ~4.8s        | ~23 min     |
| English   | ~290  | ~5.5s        | ~26 min     |
| Kannada   | ~290  | ~5.0s        | ~24 min     |
| **Total** | **~1450** | **~5.1s** | **~123 min** |


## Preprocessing_EDA.ipynb

### What This Notebook Does

Once the audio clips are collected and saved in the previous stage, the
raw data still has several problems that need to be fixed before we can
extract features from it. Clips have different loudness levels, some
have long stretches of silence at the beginning or end, and the number
of clips is not the same across all five languages. This notebook fixes
all of those problems and then produces a set of visualizations to help
us understand what the data actually looks like before we start building
models.

The notebook is split into two logical parts. The first part is
preprocessing — cleaning and organizing the data. The second part is
exploratory data analysis — understanding the data through plots and
statistics.
## Input

- `data/metadata/metadata.csv` — produced by `dataset.ipynb`
- `data/processed/{language}/*.wav` — the converted audio clips

## Notes

- This notebook must be run after `dataset.ipynb` and before
  `feature_engineering.ipynb`
- The working directory must be set to the inner project folder where
  the `data/` directory lives
- Plots are saved automatically to `data/metadata/` and do not need to
  be re-generated unless the dataset changes

  ## Next Stage

**`feature_engineering.ipynb`** — Stage 5
Extracts MFCC (39 coefficients), pitch, energy, formants F1 to F3,
spectral features, and phonotactic ratios from every clip in
`metadata_clean.csv` and produces `features_raw.csv` and
`features_selected.csv`.
#### The extracted features csv files are used by member 2 for model training(ML Pipeline)
