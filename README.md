# SHL Grammar Scoring Engine

Predicts a continuous **spoken-grammar score (1–5)** for short English speech clips, built for the
SHL Hiring Assessment "Grammar Scoring Engine" Kaggle competition. The competition evaluates Pearson
correlation and RMSE.

The primary deliverable is the notebook [`notebooks/grammar_scoring_engine.ipynb`](notebooks/grammar_scoring_engine.ipynb).
The same pipeline is available as plain Python modules in [`src/`](src/).

## Pipeline

```
WAV ─► librosa (16 kHz mono) ─► 55 acoustic stats (MFCC, energy, pitch, spectral, pauses) ───────┐
   └─► Whisper small.en (faster-whisper, CPU int8) ─► transcript                                  │
            ├─► text stats + spaCy POS/dependency features ──────────────────────────────────────┤
            ├─► RoBERTa-CoLA sentence acceptability (mean / min / fraction unacceptable) ────────┼─► Ridge / CatBoost
            ├─► TF-IDF (1–2 grams) ─► Ridge                                                       │
            └─► all-MiniLM-L6-v2 sentence embeddings ─► Ridge (alone and combined with handcrafted)┘
OOF predictions of all candidates ─► non-negative least-squares ensemble (weights from training OOF only)
```

* **Validation:** `KFold(n_splits=5, shuffle=True, random_state=42)`, with out-of-fold (OOF) predictions for
  every training row. All learned transforms (imputer, scaler, TF-IDF, Ridge α) are inside sklearn
  `Pipeline`s and are fit on training folds only. The ensemble's OOF score uses nested weights, so each fold
  is scored with weights fit on the other folds.
* **Model selection:** lowest OOF RMSE. The comparison is repeated on an independent split (seed 2024) as a
  check against selection bias.
* **Graceful degradation:** with no ASR backend, no sentence-transformers, or no CatBoost, the pipeline still
  runs on the remaining features (acoustic-only Ridge, sklearn `HistGradientBoosting` in place of CatBoost)
  and still writes a valid submission.

## Results

These are measured values from the executed notebook (Whisper `small.en` with beam 5 and VAD, 732 training
rows after excluding 37 out-of-rubric rows, `KFold(5, shuffle=True, random_state=42)`). Running
`src/train.py` reproduces them exactly. All predictions (OOF, in-sample and test) are clipped to [1, 5],
as in the submission.

| Model | 5-fold OOF RMSE | OOF Pearson | Fold RMSE (mean ± sd) |
|---|---|---|---|
| **NNLS ensemble (selected)** | **0.6408** | **0.7837** | 0.6403 ± 0.0299 |
| CatBoost, handcrafted (acoustic + linguistic + CoLA) | 0.6692 | 0.7522 | 0.6688 ± 0.0252 |
| Ridge, embeddings + handcrafted | 0.6744 | 0.7470 | 0.6737 ± 0.0343 |
| Ridge, handcrafted | 0.6913 | 0.7315 | 0.6912 ± 0.0163 |
| Ridge, MiniLM embeddings | 0.7728 | 0.6475 | 0.7720 ± 0.0409 |
| TF-IDF + Ridge | 0.7994 | 0.6185 | 0.7993 ± 0.0138 |
| Ridge, acoustic only | 0.8387 | 0.5617 | 0.8379 ± 0.0414 |
| Mean baseline | 1.0155 | – | 1.0146 ± 0.0470 |

* **Training RMSE (in-sample, selected ensemble refit on all 732 rows): 0.3232.** It is reported because the
  competition requires it. It is optimistic and is not a validation estimate.
* Ensemble weights, fit by NNLS on training OOF predictions only: CatBoost-handcrafted 0.49,
  Ridge emb+handcrafted 0.43, TF-IDF 0.07, Ridge-handcrafted 0.01, others 0.
* Robustness: on an independent split (seed 2024) the ensemble still ranks first, with OOF RMSE 0.6437 and
  Pearson 0.7800.
* Sensitivity: training with the 37 label-0 rows kept made the best single model worse on the 1–5 rows
  (OOF RMSE 0.6836 vs 0.6692).
* Train/test clip-length shift (test is mostly 45 s, train mostly 60 s) was checked in notebook §7.4 and
  causes no systematic bias.
* Acoustic-only fallback (`--no-asr`): CatBoost reached OOF RMSE 0.7671 and Pearson 0.6560 in an earlier run,
  before OOF clipping was added.
* Leaderboard score: not measured. The submission has not been uploaded to Kaggle.

## Data notes (found during inspection)

* All 985 WAV files are present, readable, 16 kHz, mono, PCM-16. Train totals 11.9 h (clips of 20–61 s);
  test totals 2.9 h (clips of 6–61 s).
* Train and test files share filenames (each folder is numbered from 0), but MD5 hashes show **no shared
  audio**.
* **37 training rows are labelled 0.0**, outside the 1–5 rubric. All of them, and only them, come from a
  separate `audio_5000+` file series, and no test file is from that series. Their transcripts are
  fluent-sounding but semantically incoherent. These rows are excluded from training and validation and
  reported in the notebook, which also includes a sensitivity check that trains with them kept.
* **`sample_submission.csv` does not match `test.csv`.** It has 204 rows, and only 25 of its filenames are
  test files. The submission uses the sample file's columns (`filename,label`) and takes its 216 rows, in
  order, from `test.csv`. Confirm which ID set the Kaggle grader expects before submitting.

## Setup

Requires Python 3.10–3.12. No GPU is needed; everything runs on CPU.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# CPU-only torch keeps the install small (needed for sentence-transformers):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m spacy download en_core_web_sm   # optional: POS / dependency features
```

On first use, the Whisper `small.en`, `all-MiniLM-L6-v2` and `textattack/roberta-base-CoLA` weights are
downloaded from the Hugging Face Hub (about 1 GB in total).

### Dataset

The competition data is restricted, so it is **not** included in this repository. Download it from the
Kaggle competition page after accepting the rules, for example:

```bash
kaggle competitions download -c <competition-slug>   # needs ~/.kaggle/kaggle.json (never commit it)
```

Extract it so the layout is:

```
Dataset_Final/
  train.csv  test.csv  sample_submission.csv
  train/audio_*.wav
  test/audio_*.wav
```

Or point to another location with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GSE_DATA_DIR` | `./Dataset_Final` | dataset root |
| `GSE_CACHE_DIR` | `./cache` | cached features, transcripts, embeddings |
| `GSE_USE_ASR` | `1` | notebook: `0` gives an acoustic-only run |
| `GSE_WHISPER_MODEL` | `small.en` | Whisper model size |
| `GSE_ASR_BEAM` / `GSE_ASR_VAD` | `5` / `1` | Whisper beam size / VAD silence skipping |

## Running

```bash
# 1. (slow, once) transcribe all audio; resumable, cached in cache/transcripts_*.json
python src/transcribe.py

# 2a. notebook: the main deliverable; writes submission.csv in the project root
jupyter nbconvert --to notebook --execute --inplace notebooks/grammar_scoring_engine.ipynb \
    --ExecutePreprocessor.timeout=-1

# 2b. or the same pipeline as a script
python src/train.py            # full pipeline
python src/train.py --no-asr   # acoustic-only; no model downloads needed
```

Both routes print the in-sample **training RMSE**, the 5-fold OOF RMSE and Pearson correlation, and a
validation report for `submission.csv`: column schema, row count and order, no NaN or inf values, and every
value in [1, 5].

## Repository structure

```
.
├── notebooks/grammar_scoring_engine.ipynb   # main report + pipeline
├── src/
│   ├── feature_extraction.py   # audio validation, acoustic / ASR / linguistic / embedding features (cached)
│   ├── train.py                # models, CV, ensemble, final fit, submission writer + validator (CLI)
│   └── transcribe.py           # batch Whisper transcription (CLI)
├── requirements.txt
├── .gitignore                  # excludes data, caches, weights, submission, credentials
└── README.md
```

## Limitations

* **The ASR step filters the grammar signal.** Whisper tends to produce fluent, well-formed text. It can hide
  real grammatical errors and introduce recognition errors the speaker never made.
* **Proxy features.** Linguistic features measure complexity more than correctness. Acoustic features
  measure fluency and recording conditions, and acoustic fluency is not grammatical accuracy.
* **Small data.** With about 730 usable labels, differences between models smaller than the fold-to-fold
  standard deviation are not meaningful.
* **Data quirks.** The excluded label-0 rows and the mismatched sample submission are described above.

## Reproducibility

Fixed seeds (`42` for folds and models), a pinned scikit-learn version, and cached intermediate features.
CatBoost and the BLAS backend may differ slightly between platforms. Whisper int8 CPU decoding is
deterministic for a fixed model and configuration.

> scikit-learn is pinned to 1.5.2 because Windows Smart App Control blocked the DLLs of a newer wheel on the
> development machine.

## Data use

Competition data, transcripts derived from it, caches and model weights are all git-ignored. Do not upload
them. Follow the competition's rules on data use.
