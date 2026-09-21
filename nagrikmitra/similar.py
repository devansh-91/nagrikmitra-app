"""
Similar-ticket lookup via TF-IDF cosine similarity against data/complaints.csv.

Responsibilities:
- Build/cache a TF-IDF matrix over historical complaints at import/startup.
- similar_tickets(text: str, top_k: int = 5) -> list[dict] (id, text, domain, score)
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nagrikmitra.config import COMPLAINTS_CSV_PATH
from nagrikmitra.preprocess import normalize

_corpus_df = None
_vectorizer = None
_matrix = None


def _load():
    global _corpus_df, _vectorizer, _matrix
    if _corpus_df is not None:
        return

    df = pd.read_csv(COMPLAINTS_CSV_PATH)
    df["clean_text"] = df["text"].astype(str).apply(normalize)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(df["clean_text"])

    _corpus_df = df
    _vectorizer = vectorizer
    _matrix = matrix


def similar_tickets(text: str, top_k: int = 5) -> list:
    """Return the top_k historical complaints most similar to `text` by
    TF-IDF cosine similarity, as [{id, text, domain, score}, ...]."""
    _load()

    clean_text = normalize(text)
    query_vec = _vectorizer.transform([clean_text])
    scores = cosine_similarity(query_vec, _matrix)[0]

    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        row = _corpus_df.iloc[idx]
        results.append(
            {
                "id": int(row["id"]),
                "text": str(row["text"]),
                "domain": str(row["domain"]),
                "score": round(float(scores[idx]), 4),
            }
        )
    return results
