"""Feature extraction for the SHL Grammar Scoring Engine.

Three feature families, each cached to disk so expensive steps run once:
  * acoustic   - librosa statistics (duration, MFCC, energy, pitch, pauses ...)
  * transcript - optional Whisper ASR (faster-whisper preferred, openai-whisper fallback)
  * linguistic - text statistics computed from transcripts (+ optional spaCy POS,
                 optional sentence-transformer embeddings)

Every optional dependency is imported lazily; when one is missing the function
logs why and returns None / empty features so the rest of the pipeline still runs.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("GSE_DATA_DIR", PROJECT_ROOT / "Dataset_Final"))
CACHE_DIR = Path(os.environ.get("GSE_CACHE_DIR", PROJECT_ROOT / "cache"))
TARGET_SR = 16_000  # Whisper and most speech models expect 16 kHz mono
SEED = 42


def audio_path(split: str, filename: str) -> Path:
    return DATA_DIR / split / filename


# ---------------------------------------------------------------------------
# Audio loading / validation
# ---------------------------------------------------------------------------
def load_audio(path: Path, sr: int = TARGET_SR) -> np.ndarray:
    """Load audio as mono float32, resampled to `sr`. Raises on unreadable files."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")
    return y.astype(np.float32)


def audio_metadata(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Header-level metadata for every row (no decoding). Records errors instead of dropping rows."""
    import soundfile as sf

    rows = []
    for fn in df["filename"]:
        p = audio_path(split, fn)
        rec = {"filename": fn, "exists": p.exists(), "orig_sr": np.nan, "channels": np.nan,
               "duration_s": np.nan, "subtype": None, "error": None}
        if rec["exists"]:
            try:
                info = sf.info(str(p))
                rec.update(orig_sr=info.samplerate, channels=info.channels,
                           duration_s=info.frames / info.samplerate, subtype=info.subtype)
            except Exception as e:  # corrupt / unsupported file
                rec["error"] = f"{type(e).__name__}: {e}"
        else:
            rec["error"] = "missing file"
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Acoustic features
# ---------------------------------------------------------------------------
def _stats(prefix: str, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_std": float(x.std())}


def acoustic_features(y: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Fixed-length acoustic summary of one clip."""
    import librosa

    f = {"duration_s": len(y) / sr}
    hop = 512

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    for i in range(mfcc.shape[0]):
        f.update(_stats(f"mfcc{i}", mfcc[i]))
    f.update(_stats("mfcc_delta_abs", np.abs(librosa.feature.delta(mfcc))))

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    f.update(_stats("rms", rms))
    f.update(_stats("zcr", librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]))
    f.update(_stats("spec_centroid", librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]))
    f.update(_stats("spec_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]))
    f.update(_stats("spec_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    for i in range(contrast.shape[0]):
        f[f"spec_contrast{i}_mean"] = float(contrast[i].mean())

    # Pitch (YIN is fast; keep only frames with energy so silence doesn't pollute stats)
    try:
        f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr, hop_length=hop)
        voiced = f0[: len(rms)][rms[: len(f0)] > 0.5 * np.median(rms)]
        f.update(_stats("f0", voiced))
        f["f0_range"] = float(np.ptp(voiced)) if voiced.size else np.nan
    except Exception:
        f.update({"f0_mean": np.nan, "f0_std": np.nan, "f0_range": np.nan})

    # Silence / pause statistics from an energy threshold relative to the clip's own level
    frame_s = hop / sr
    db = librosa.amplitude_to_db(rms, ref=np.max)
    silent = db < -35
    f["silence_ratio"] = float(silent.mean())
    runs, run = [], 0
    for s in silent:
        if s:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    pauses = np.array([r * frame_s for r in runs if r * frame_s >= 0.3])  # internal pauses >= 300 ms
    f["n_pauses"] = int(pauses.size)
    f["pause_rate_per_min"] = pauses.size / max(f["duration_s"] / 60, 1e-6)
    f["mean_pause_s"] = float(pauses.mean()) if pauses.size else 0.0
    f["max_pause_s"] = float(pauses.max()) if pauses.size else 0.0
    f["speech_s"] = float((~silent).sum() * frame_s)
    return f


def extract_acoustic(df: pd.DataFrame, split: str, cache: bool = True) -> pd.DataFrame:
    """Acoustic features for every row of df; cached in CACHE_DIR/acoustic_<split>.csv.

    Failed files keep their row (features NaN, `audio_error` set) so nothing is dropped silently.
    """
    path = CACHE_DIR / f"acoustic_{split}.csv"
    cached = pd.read_csv(path) if cache and path.exists() else pd.DataFrame(columns=["filename"])
    done = set(cached["filename"])
    todo = [fn for fn in df["filename"] if fn not in done]

    rows = []
    for i, fn in enumerate(todo, 1):
        rec = {"filename": fn, "audio_error": None}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rec.update(acoustic_features(load_audio(audio_path(split, fn))))
        except Exception as e:
            rec["audio_error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
        if i % 100 == 0:
            print(f"  acoustic[{split}] {i}/{len(todo)}")
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.to_csv(path, index=False)
    return df[["filename"]].merge(cached, on="filename", how="left")


# ---------------------------------------------------------------------------
# ASR transcripts (optional)
# ---------------------------------------------------------------------------
# No `initial_prompt`: in testing, Whisper regurgitated prompt text into transcripts
# (fabricated words), which is worse than its tendency to drop some fillers.
ASR_BEAM = int(os.environ.get("GSE_ASR_BEAM", "5"))
# Silero VAD skips silences: ~20% faster here, and fewer truncated/hallucinated outputs (faster-whisper only)
ASR_VAD = os.environ.get("GSE_ASR_VAD", "1") == "1"


def _load_asr(model_size: str):
    """Return (backend_name, transcribe_fn) or (None, None) if no ASR backend is usable."""
    try:
        from faster_whisper import WhisperModel

        m = WhisperModel(model_size, device="cpu", compute_type="int8",
                         cpu_threads=max(1, (os.cpu_count() or 8) // 2))  # physical cores; SMT didn't help

        def run(y):
            segs, _ = m.transcribe(y, language="en", beam_size=ASR_BEAM, vad_filter=ASR_VAD,
                                   condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segs).strip()

        return f"faster-whisper/{model_size}", run
    except Exception as e:
        print(f"faster-whisper unavailable ({type(e).__name__}: {e}); trying openai-whisper")
    try:
        import whisper

        m = whisper.load_model(model_size)

        def run(y):
            r = m.transcribe(y, language="en", fp16=False, beam_size=ASR_BEAM, condition_on_previous_text=False)
            return r["text"].strip()

        return f"openai-whisper/{model_size}", run
    except Exception as e:
        print(f"openai-whisper unavailable ({type(e).__name__}: {e}); transcripts disabled")
    return None, None


def transcribe(df: pd.DataFrame, split: str, model_size: str | None = None) -> pd.DataFrame | None:
    """Transcribe every row, caching incrementally to CACHE_DIR/transcripts_<split>_<model>.json.

    Returns a DataFrame [filename, transcript, asr_error] or None if no ASR backend exists
    and nothing is cached. Failures yield an empty transcript plus an error message - never
    a made-up transcript.
    """
    model_size = model_size or os.environ.get("GSE_WHISPER_MODEL", "small.en")
    path = CACHE_DIR / f"transcripts_{split}_{model_size}_b{ASR_BEAM}{'_vad' if ASR_VAD else ''}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [fn for fn in df["filename"] if fn not in cache]

    if todo:
        backend, run = _load_asr(model_size)
        if run is None:
            if not cache:
                return None
            print(f"WARNING: {len(todo)} files untranscribed and no ASR backend available")
        else:
            print(f"Transcribing {len(todo)} {split} files with {backend} (cached: {len(cache)})")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            for i, fn in enumerate(todo, 1):
                try:
                    cache[fn] = {"text": run(load_audio(audio_path(split, fn))), "error": None}
                except Exception as e:
                    cache[fn] = {"text": "", "error": f"{type(e).__name__}: {e}"}
                if i % 25 == 0 or i == len(todo):
                    path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
                    print(f"  asr[{split}] {i}/{len(todo)}")

    return pd.DataFrame({
        "filename": df["filename"].values,
        "transcript": [cache.get(fn, {}).get("text", "") for fn in df["filename"]],
        "asr_error": [cache.get(fn, {"error": "not transcribed"}).get("error") for fn in df["filename"]],
    })


# ---------------------------------------------------------------------------
# Linguistic features
# ---------------------------------------------------------------------------
FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm", "umm", "like", "so", "well"}
SUBORDINATORS = {"because", "although", "though", "while", "whereas", "since", "unless", "if",
                 "when", "whenever", "which", "who", "whom", "whose", "that", "where", "after",
                 "before", "until", "so that"}
_WORD = re.compile(r"[a-zA-Z']+")
_SENT = re.compile(r"[.!?]+")

_nlp = None  # spaCy pipeline, loaded once on demand


def _spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception as e:
            print(f"spaCy POS features disabled ({type(e).__name__}: {e})")
            _nlp = False
    return _nlp or None


def linguistic_features(text: str, duration_s: float | None = None) -> dict:
    text = (text or "").strip()
    words = [w.lower() for w in _WORD.findall(text)]
    n = len(words)
    sents = [s for s in _SENT.split(text) if _WORD.search(s)]
    sent_lens = np.array([len(_WORD.findall(s)) for s in sents]) if sents else np.array([n])
    wl = np.array([len(w) for w in words]) if words else np.array([0])
    cnt = Counter(words)

    f = {
        "empty_transcript": int(n == 0),
        "n_words": n,
        "n_sentences": len(sents),
        "sent_len_mean": float(sent_lens.mean()),
        "sent_len_var": float(sent_lens.var()),
        "sent_len_max": float(sent_lens.max()),
        "ttr": len(cnt) / n if n else 0.0,
        # root TTR is less length-sensitive than plain TTR
        "root_ttr": len(cnt) / np.sqrt(n) if n else 0.0,
        "word_len_mean": float(wl.mean()),
        "word_len_std": float(wl.std()),
        "long_word_ratio": float((wl >= 7).mean()) if n else 0.0,
        "filler_ratio": sum(cnt[w] for w in FILLERS) / n if n else 0.0,
        "immediate_repeats": sum(a == b for a, b in zip(words, words[1:])),
        "subordinator_ratio": sum(cnt[w] for w in SUBORDINATORS) / n if n else 0.0,
        "pronoun_i_ratio": cnt["i"] / n if n else 0.0,
        "comma_per_word": text.count(",") / n if n else 0.0,
    }
    if duration_s and duration_s > 0:
        f["words_per_min"] = n / (duration_s / 60)

    nlp = _spacy()
    if nlp is not None and n:
        doc = nlp(text)
        toks = [t for t in doc if not t.is_punct and not t.is_space]
        pos = Counter(t.pos_ for t in toks)
        tot = max(len(toks), 1)
        for tag in ("NOUN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ"):
            f[f"pos_{tag.lower()}"] = pos[tag] / tot
        deps = Counter(t.dep_ for t in toks)
        f["dep_clausal"] = sum(deps[d] for d in ("advcl", "ccomp", "xcomp", "acl", "relcl")) / tot
        depths = []
        for t in toks:
            d = 0
            while t.head.i != t.i and d < 50:  # root's head is itself
                t, d = t.head, d + 1
            depths.append(d)
        f["tree_depth_mean"] = float(np.mean(depths))
        f["verbs_per_sentence"] = (pos["VERB"] + pos["AUX"]) / max(len(list(doc.sents)), 1)
    return f


def extract_linguistic(transcripts: pd.DataFrame, durations: pd.Series | None = None) -> pd.DataFrame:
    durs = durations.values if durations is not None else [None] * len(transcripts)
    rows = [linguistic_features(t, d) for t, d in zip(transcripts["transcript"], durs)]
    out = pd.DataFrame(rows)
    out.insert(0, "filename", transcripts["filename"].values)
    return out


# ---------------------------------------------------------------------------
# Grammatical acceptability (optional): RoBERTa fine-tuned on CoLA scores each
# transcript sentence; this is the feature closest to "grammar" as such.
# ---------------------------------------------------------------------------
COLA_MODEL = "textattack/roberta-base-CoLA"  # LABEL_1 = acceptable (verified on hand-made examples)


def acceptability_features(filenames: list[str], texts: list[str], split: str) -> pd.DataFrame | None:
    """Per-clip stats of sentence-level P(acceptable); cached per filename+text in CACHE_DIR/cola_<split>.json."""
    path = CACHE_DIR / f"cola_{split}.json"
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    keys = [f"{fn}\t{t or ''}" for fn, t in zip(filenames, texts)]
    todo = [i for i, k in enumerate(keys) if k not in store]
    if todo:
        try:
            from transformers import pipeline

            clf = pipeline("text-classification", model=COLA_MODEL, device=-1)
        except Exception as ex:
            print(f"acceptability features disabled ({type(ex).__name__}: {ex})")
            return None
        for i in todo:
            sents = [s.strip() for s in _SENT.split(texts[i] or "") if _WORD.search(s)]
            probs = []
            if sents:
                out = clf(sents, top_k=None, truncation=True, max_length=128, batch_size=16)
                probs = [next(d["score"] for d in o if d["label"] == "LABEL_1") for o in out]
            store[keys[i]] = probs
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store), encoding="utf-8")

    rows = []
    for k in keys:
        p = np.array(store[k])
        rows.append({"cola_mean": p.mean() if p.size else np.nan, "cola_min": p.min() if p.size else np.nan,
                     "cola_frac_unacceptable": (p < 0.5).mean() if p.size else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sentence embeddings (optional)
# ---------------------------------------------------------------------------
def sentence_embeddings(filenames: list[str], texts: list[str], split: str,
                        model: str = "all-MiniLM-L6-v2") -> np.ndarray | None:
    """Normalised sentence-transformer embeddings, cached per filename (and transcript text)
    in CACHE_DIR/emb_<split>_<model>.npz. Returns rows in the order of `filenames`."""
    path = CACHE_DIR / f"emb_{split}_{model.replace('/', '_')}.npz"
    keys = [f"{fn}\t{t or ''}" for fn, t in zip(filenames, texts)]  # text in key: stale if transcript changes
    store = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        store = dict(zip(z["keys"].tolist(), z["emb"]))
    todo = [i for i, k in enumerate(keys) if k not in store]
    if todo:
        try:
            from sentence_transformers import SentenceTransformer

            enc = SentenceTransformer(model, device="cpu").encode(
                [texts[i] or "" for i in todo], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        except Exception as ex:
            print(f"sentence embeddings disabled ({type(ex).__name__}: {ex})")
            return None
        store.update({keys[i]: e for i, e in zip(todo, enc)})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=np.array(list(store)), emb=np.stack(list(store.values())))
    return np.stack([store[k] for k in keys])
