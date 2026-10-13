"""Model training, cross-validation, model selection and submission for the Grammar Scoring Engine.

Usable from the notebook (import functions) or the command line:

    python src/train.py              # full pipeline incl. Whisper ASR (cached after first run)
    python src/train.py --no-asr     # acoustic-only pipeline, no model downloads needed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_extraction as fx  # noqa: E402

SEED = fx.SEED
N_FOLDS = 5
ALPHAS = np.logspace(-2, 4, 25)
SCORE_RANGE = (1.0, 5.0)  # competition rubric


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(y, p) -> float:
    try:
        from sklearn.metrics import root_mean_squared_error

        return float(root_mean_squared_error(y, p))
    except ImportError:  # scikit-learn < 1.4
        return float(np.sqrt(mean_squared_error(y, p)))


def pearson(y, p) -> float:
    """Pearson r; NaN (not an exception) when either side has zero variance."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(pearsonr(y, p)[0])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(fx.DATA_DIR / "train.csv")
    test = pd.read_csv(fx.DATA_DIR / "test.csv")
    sample = pd.read_csv(fx.DATA_DIR / "sample_submission.csv")
    return train, test, sample


def split_out_of_rubric(train: pd.DataFrame, lo: float = SCORE_RANGE[0], hi: float = SCORE_RANGE[1]):
    """Separate rows whose label is missing or outside the 1-5 rubric. Returns (kept, excluded)."""
    bad = train["label"].isna() | (train["label"] < lo) | (train["label"] > hi)
    return train[~bad].reset_index(drop=True), train[bad].reset_index(drop=True)


def build_features(train: pd.DataFrame, test: pd.DataFrame, use_asr: bool = True) -> dict:
    """Assemble every feature block for train and test. Optional blocks are None when unavailable."""
    ac_tr, ac_te = fx.extract_acoustic(train, "train"), fx.extract_acoustic(test, "test")
    acoustic_cols = [c for c in ac_tr.columns if c not in ("filename", "audio_error")]
    feats = {"acoustic": (ac_tr[acoustic_cols], ac_te[acoustic_cols]), "audio_errors": (ac_tr, ac_te),
             "linguistic": None, "text": None, "embedding": None, "transcripts": None}

    tr_tx = fx.transcribe(train, "train") if use_asr else None
    te_tx = fx.transcribe(test, "test") if (use_asr and tr_tx is not None) else None
    if tr_tx is None or te_tx is None:
        print("Transcripts unavailable -> text-based models skipped")
        return feats

    feats["transcripts"] = (tr_tx, te_tx)
    feats["text"] = (tr_tx["transcript"].fillna(""), te_tx["transcript"].fillna(""))
    li_tr = fx.extract_linguistic(tr_tx, ac_tr["duration_s"]).drop(columns="filename")
    li_te = fx.extract_linguistic(te_tx, ac_te["duration_s"]).drop(columns="filename")
    co_tr = fx.acceptability_features(train["filename"].tolist(), feats["text"][0].tolist(), "train")
    co_te = (fx.acceptability_features(test["filename"].tolist(), feats["text"][1].tolist(), "test")
             if co_tr is not None else None)
    if co_tr is not None and co_te is not None:
        li_tr, li_te = _hstack(li_tr, co_tr), _hstack(li_te, co_te)
    li_te = li_te.reindex(columns=li_tr.columns)  # identical column set/order
    feats["linguistic"] = (li_tr, li_te)

    e_tr = fx.sentence_embeddings(train["filename"].tolist(), feats["text"][0].tolist(), "train")
    e_te = (fx.sentence_embeddings(test["filename"].tolist(), feats["text"][1].tolist(), "test")
            if e_tr is not None else None)
    if e_tr is not None and e_te is not None:
        feats["embedding"] = (e_tr, e_te)
    return feats


# ---------------------------------------------------------------------------
# Candidate models. Every learned transform lives inside the Pipeline, so in CV it
# is fit on the training folds only (no leakage into validation folds).
# ---------------------------------------------------------------------------
def _tabular_ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS))


def _gbm():
    try:
        from catboost import CatBoostRegressor

        return "catboost", CatBoostRegressor(iterations=800, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                                             loss_function="RMSE", random_seed=SEED, verbose=0,
                                             thread_count=-1, allow_writing_files=False)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return "hgb", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_depth=4,
                                                     l2_regularization=1.0, random_state=SEED)


def _hstack(*blocks):
    return pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)


def candidate_models(feats: dict) -> dict:
    """name -> (X_train, X_test, estimator). Only models whose inputs exist are returned."""
    ac_tr, ac_te = feats["acoustic"]
    m = {"ridge_acoustic": (ac_tr, ac_te, _tabular_ridge())}
    gbm_name, gbm = _gbm()

    if feats["linguistic"] is None:
        m[f"{gbm_name}_acoustic"] = (ac_tr, ac_te, gbm)
        return m

    li_tr, li_te = feats["linguistic"]
    hc_tr, hc_te = _hstack(ac_tr, li_tr), _hstack(ac_te, li_te)
    m["ridge_handcrafted"] = (hc_tr, hc_te, _tabular_ridge())
    m[f"{gbm_name}_handcrafted"] = (hc_tr, hc_te, gbm)

    tx_tr, tx_te = feats["text"]
    tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("ridge", RidgeCV(alphas=ALPHAS)),
    ])
    m["tfidf_ridge"] = (tx_tr, tx_te, tfidf)

    if feats["embedding"] is not None:
        e_tr, e_te = feats["embedding"]
        m["embedding_ridge"] = (e_tr, e_te, make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)))
        # embeddings + handcrafted in one regularised linear model
        cols_e = [f"emb{i}" for i in range(e_tr.shape[1])]
        eh_tr = _hstack(hc_tr, pd.DataFrame(e_tr, columns=cols_e))
        eh_te = _hstack(hc_te, pd.DataFrame(e_te, columns=cols_e))
        prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                                   list(eh_tr.columns))])
        m["ridge_emb+handcrafted"] = (eh_tr, eh_te, make_pipeline(prep, RidgeCV(alphas=ALPHAS)))
    return m


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def make_folds(n: int, seed: int = SEED) -> list:
    return list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n)))


def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def cross_validate(X, y: np.ndarray, est, folds) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold predictions + per-fold RMSE / Pearson.

    Predictions are clipped to SCORE_RANGE exactly as the submission is, so validation
    metrics describe the predictions we actually submit.
    """
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, va) in enumerate(folds):
        model = clone(est).fit(_take(X, tr), y[tr])
        oof[va] = np.clip(model.predict(_take(X, va)), *SCORE_RANGE)
        rows.append({"fold": k + 1, "rmse": rmse(y[va], oof[va]), "pearson": pearson(y[va], oof[va])})
    return oof, pd.DataFrame(rows)


def mean_baseline_cv(y, folds):
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, va) in enumerate(folds):
        oof[va] = y[tr].mean()
        rows.append({"fold": k + 1, "rmse": rmse(y[va], oof[va]), "pearson": float("nan")})
    return oof, pd.DataFrame(rows)


def ensemble_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least-squares weights on OOF predictions, normalised to sum to 1."""
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def nested_ensemble_oof(P: np.ndarray, y: np.ndarray, folds) -> tuple[np.ndarray, pd.DataFrame]:
    """Honest ensemble estimate: fold k's weights come only from the OOF rows of the other folds."""
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, va) in enumerate(folds):
        w = ensemble_weights(P[tr], y[tr])
        oof[va] = np.clip(P[va] @ w, *SCORE_RANGE)
        rows.append({"fold": k + 1, "rmse": rmse(y[va], oof[va]), "pearson": pearson(y[va], oof[va])})
    return oof, pd.DataFrame(rows)


def summarize(name, y, oof, fm) -> dict:
    # The mean baseline is constant within each fold, so its pooled Pearson is meaningless -> NaN.
    r = float("nan") if name == "mean_baseline" else pearson(y, oof)
    return {"model": name, "oof_rmse": rmse(y, oof), "oof_pearson": r,
            "fold_rmse_mean": fm["rmse"].mean(), "fold_rmse_std": fm["rmse"].std(ddof=1),
            "fold_pearson_mean": fm["pearson"].mean(), "fold_pearson_std": fm["pearson"].std(ddof=1)}


def run_cv(models: dict, y: np.ndarray, folds) -> dict:
    """CV for the mean baseline, every candidate, and a NNLS ensemble of the candidates."""
    oofs, fold_metrics, rows = {}, {}, []
    oofs["mean_baseline"], fold_metrics["mean_baseline"] = mean_baseline_cv(y, folds)
    for name, (Xtr, _, est) in models.items():
        print(f"CV: {name}")
        oofs[name], fold_metrics[name] = cross_validate(Xtr, y, est, folds)

    names = list(models)
    if len(names) > 1:
        P = np.column_stack([oofs[n] for n in names])
        oofs["ensemble"], fold_metrics["ensemble"] = nested_ensemble_oof(P, y, folds)
    for n in oofs:
        rows.append(summarize(n, y, oofs[n], fold_metrics[n]))
    table = pd.DataFrame(rows).sort_values("oof_rmse").reset_index(drop=True)
    return {"table": table, "oof": oofs, "folds": fold_metrics, "members": names}


# ---------------------------------------------------------------------------
# Final fit + submission
# ---------------------------------------------------------------------------
def fit_predict(selected: str, models: dict, cv: dict, y: np.ndarray):
    """Refit the selected model (or every ensemble member) on all training data.

    Returns (train_pred_in_sample, test_pred, ensemble_weights_or_None), both clipped to
    SCORE_RANGE like the OOF predictions, so training and validation metrics are comparable.
    """
    tr_pred, te_pred, w = _fit_predict_raw(selected, models, cv, y)
    clip = lambda p: None if p is None else np.clip(p, *SCORE_RANGE)
    return clip(tr_pred), clip(te_pred), w


def _fit_predict_raw(selected: str, models: dict, cv: dict, y: np.ndarray):
    if selected == "ensemble":
        names = cv["members"]
        w = ensemble_weights(np.column_stack([cv["oof"][n] for n in names]), y)
        tr_cols, te_cols = [], []
        for n in names:
            Xtr, Xte, est = models[n]
            m = clone(est).fit(Xtr, y)
            tr_cols.append(m.predict(Xtr))
            te_cols.append(m.predict(Xte))
        return np.column_stack(tr_cols) @ w, np.column_stack(te_cols) @ w, dict(zip(names, map(float, w)))
    if selected == "mean_baseline":
        return np.full(len(y), y.mean()), None, None
    Xtr, Xte, est = models[selected]
    m = clone(est).fit(Xtr, y)
    return m.predict(Xtr), m.predict(Xte), None


def write_submission(test: pd.DataFrame, sample: pd.DataFrame, preds: np.ndarray, path: Path,
                     clip: tuple[float, float]) -> pd.DataFrame:
    """Build submission with sample_submission's column names/order and test.csv's rows/order."""
    id_col, target_col = sample.columns[0], sample.columns[1]
    preds = np.clip(np.asarray(preds, float), *clip)
    sub = pd.DataFrame({id_col: test["filename"].values, target_col: preds})[list(sample.columns)]
    sub.to_csv(path, index=False)
    return sub


def validate_submission(sub: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, clip) -> dict:
    target = sample.columns[1]
    v = sub[target].to_numpy(float)
    report = {
        "columns_match_sample": list(sub.columns) == list(sample.columns),
        "rows": len(sub), "expected_rows(test.csv)": len(test),
        "row_order_matches_test": sub.iloc[:, 0].tolist() == test["filename"].tolist(),
        "unique_ids": sub.iloc[:, 0].is_unique,
        "missing_or_inf": int((~np.isfinite(v)).sum()),
        "min": float(v.min()), "max": float(v.max()),
        "within_range": bool(((v >= clip[0]) & (v <= clip[1])).all()),
        "sample_rows": len(sample),
        "sample_ids_in_test": int(sample.iloc[:, 0].isin(test["filename"]).sum()),
    }
    report["ok"] = all([report["columns_match_sample"], report["rows"] == len(test),
                        report["row_order_matches_test"], report["unique_ids"],
                        report["missing_or_inf"] == 0, report["within_range"]])
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-asr", action="store_true", help="skip Whisper/text features")
    ap.add_argument("--out", default=str(fx.PROJECT_ROOT / "submission.csv"))
    args = ap.parse_args(argv)

    np.random.seed(SEED)
    train, test, sample = load_data()
    train, excluded = split_out_of_rubric(train)
    print(f"Excluded {len(excluded)} rows with labels outside {SCORE_RANGE}: "
          f"{excluded['label'].value_counts().to_dict()}")
    y = train["label"].to_numpy(float)
    feats = build_features(train, test, use_asr=not args.no_asr)
    models = candidate_models(feats)
    folds = make_folds(len(y))
    cv = run_cv(models, y, folds)
    print(cv["table"].to_string(index=False))

    selected = cv["table"].iloc[0]["model"]
    train_pred, test_pred, w = fit_predict(selected, models, cv, y)
    clip = SCORE_RANGE
    print(f"\nSelected: {selected}  weights={w}")
    print(f"Training RMSE (in-sample, NOT validation): {rmse(y, train_pred):.4f}")
    row = cv["table"].set_index("model").loc[selected]
    print(f"5-fold OOF RMSE: {row.oof_rmse:.4f}  OOF Pearson: {row.oof_pearson:.4f}")

    sub = write_submission(test, sample, test_pred, Path(args.out), clip)
    print("Submission:", Path(args.out).resolve())
    print(validate_submission(sub, test, sample, clip))


if __name__ == "__main__":
    main()
