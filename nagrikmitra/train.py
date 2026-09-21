"""
Train sklearn models: TF-IDF(1-2 grams) + LogisticRegression, two heads
(domain, priority). Stratified 70/15/15 split by domain.

Responsibilities:
- Fit models/domain.joblib and models/priority.joblib.
- Write reports/metrics.json: accuracy + macro-F1 per head, keyword-baseline
  domain accuracy, language-sliced metrics (en/hi/hinglish), high-priority
  recall, and a disclaimer about synthetic-template memorization.
- Never overwrite metrics.json with fabricated/rounded 100% scores.

Run: python -m nagrikmitra.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nagrikmitra.config import (
    COMPLAINTS_CSV_PATH,
    DOMAIN_MODEL_PATH,
    METRICS_JSON_PATH,
    PRIORITY_MODEL_PATH,
)
from nagrikmitra.preprocess import normalize
from nagrikmitra.rules import predict_domain_keyword

SEED = 42


def _build_pipeline(C: float = 3.0) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=SEED, C=C),
            ),
        ]
    )


def _split(df: pd.DataFrame):
    # 70/15/15 stratified by domain: first split off 30% (val+test), then
    # split that 30% in half, stratified by domain both times.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["domain"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["domain"]
    )
    return train_df, val_df, test_df


def train() -> dict:
    df = pd.read_csv(COMPLAINTS_CSV_PATH)
    df["clean_text"] = df["text"].astype(str).apply(normalize)

    train_df, val_df, test_df = _split(df)

    # --- Domain head ---
    domain_pipeline = _build_pipeline(C=3.0)
    domain_pipeline.fit(train_df["clean_text"], train_df["domain"])
    domain_pred = domain_pipeline.predict(test_df["clean_text"])
    domain_accuracy = accuracy_score(test_df["domain"], domain_pred)
    domain_macro_f1 = f1_score(test_df["domain"], domain_pred, average="macro")

    # --- Priority head ---
    priority_pipeline = _build_pipeline(C=2.0)
    priority_pipeline.fit(train_df["clean_text"], train_df["priority"])
    priority_pred = priority_pipeline.predict(test_df["clean_text"])
    priority_accuracy = accuracy_score(test_df["priority"], priority_pred)
    priority_macro_f1 = f1_score(test_df["priority"], priority_pred, average="macro")

    # --- Keyword baseline (domain only) ---
    keyword_pred = test_df["clean_text"].apply(predict_domain_keyword)
    keyword_accuracy = accuracy_score(test_df["domain"], keyword_pred)

    # --- Language-sliced metrics ---
    language_slices = {}
    for lang in ["en", "hi", "hinglish"]:
        mask = test_df["language"] == lang
        if mask.sum() == 0:
            language_slices[lang] = {"domain_accuracy": None, "priority_accuracy": None}
            continue
        lang_domain_acc = accuracy_score(
            test_df.loc[mask, "domain"], domain_pred[mask.values]
        )
        lang_priority_acc = accuracy_score(
            test_df.loc[mask, "priority"], priority_pred[mask.values]
        )
        language_slices[lang] = {
            "domain_accuracy": round(float(lang_domain_acc), 4),
            "priority_accuracy": round(float(lang_priority_acc), 4),
        }

    # --- High-priority recall ---
    high_recall = recall_score(
        test_df["priority"], priority_pred, labels=["High"], average="macro"
    )

    metrics = {
        "_comment": "Populated by nagrikmitra/train.py. Real numbers from a held-out test split.",
        "domain_head": {
            "test_accuracy": round(float(domain_accuracy), 4),
            "test_macro_f1": round(float(domain_macro_f1), 4),
        },
        "priority_head": {
            "test_accuracy": round(float(priority_accuracy), 4),
            "test_macro_f1": round(float(priority_macro_f1), 4),
        },
        "keyword_baseline": {
            "domain_accuracy": round(float(keyword_accuracy), 4),
        },
        "language_slices": language_slices,
        "high_priority_recall": round(float(high_recall), 4),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "disclaimer": (
            "Synthetic complaint templates tend to be memorized; treat these "
            "numbers as indicative, not a real-world benchmark. Never quote "
            "100% as a production F1."
        ),
    }

    DOMAIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(domain_pipeline, DOMAIN_MODEL_PATH)
    joblib.dump(priority_pipeline, PRIORITY_MODEL_PATH)

    METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    return metrics


def main() -> None:
    metrics = train()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
