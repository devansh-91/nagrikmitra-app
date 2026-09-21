"""
Lightweight explainability for the UI/API (why this domain/priority).

Responsibilities:
- top_features(vectorizer, model, text, k=5) -> list[(token, weight)] per head,
  using the linear model's coefficients on the TF-IDF vector of this ticket.
- Used to populate the `explain` field in POST /predict responses.
"""


def top_features(pipeline, text: str, k: int = 5) -> list:
    """Return the top-k TF-IDF tokens (by |coefficient * tfidf weight|) that
    pushed the linear model toward its predicted class for this ticket."""
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    x = vectorizer.transform([text])
    predicted_label = pipeline.predict([text])[0]
    class_idx = list(clf.classes_).index(predicted_label)

    # Binary LogisticRegression stores a single coefficient row; multiclass
    # (multinomial) stores one row per class.
    coef_row = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_[class_idx]

    feature_names = vectorizer.get_feature_names_out()
    x_coo = x.tocoo()

    contributions = []
    for col, value in zip(x_coo.col, x_coo.data):
        weight = float(coef_row[col]) * float(value)
        contributions.append((feature_names[col], round(weight, 4)))

    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return contributions[:k]


def explain_prediction(domain_pipeline, priority_pipeline, clean_text: str, k: int = 5) -> dict:
    """Explain both heads' predictions for a preprocessed ticket text."""
    return {
        "domain": top_features(domain_pipeline, clean_text, k=k),
        "priority": top_features(priority_pipeline, clean_text, k=k),
    }
