"""
Inference: the routing engine. sklearn always runs here; this is the only
module allowed to decide domain/priority/needs_human.

Responsibilities:
- Load models/domain.joblib, models/priority.joblib (lazy singleton).
- predict(text, channel="web") -> dict with:
    language, domain, department, priority, confidence, needs_human,
    sla_hours, domain_top3, priority_top3
- confidence = min(domain_top1_proba, priority_top1_proba)
- needs_human = True if language == "other" OR confidence < 0.45
    OR (domain_top1_proba - domain_top2_proba) < 0.08
  -> in that case domain is forced to "Other / Unknown".
"""
import joblib

from nagrikmitra.config import (
    CONFIDENCE_ABSTAIN_THRESHOLD,
    DOMAIN_MARGIN_THRESHOLD,
    DOMAIN_MODEL_PATH,
    PRIORITY_MODEL_PATH,
)
from nagrikmitra.language import detect_language
from nagrikmitra.preprocess import normalize
from nagrikmitra.taxonomy import OTHER_DOMAIN, get_department, get_sla_hours

_domain_model = None
_priority_model = None


def load_models():
    """Lazily load and cache (domain_pipeline, priority_pipeline)."""
    global _domain_model, _priority_model
    if _domain_model is None:
        _domain_model = joblib.load(DOMAIN_MODEL_PATH)
    if _priority_model is None:
        _priority_model = joblib.load(PRIORITY_MODEL_PATH)
    return _domain_model, _priority_model


def _top_k(model, clean_text: str, k: int = 3) -> list:
    """Return [(label, proba), ...] sorted descending, top k."""
    proba = model.predict_proba([clean_text])[0]
    classes = model.classes_
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    return [(str(label), round(float(p), 4)) for label, p in ranked[:k]]


def predict(text: str, channel: str = "web") -> dict:
    """Run the full routing pipeline for one complaint. sklearn always runs
    and is the sole authority on domain/priority/needs_human."""
    domain_model, priority_model = load_models()

    language = detect_language(text)
    clean_text = normalize(text)

    domain_top3 = _top_k(domain_model, clean_text, k=3)
    priority_top3 = _top_k(priority_model, clean_text, k=3)

    domain_top1_label, domain_top1_proba = domain_top3[0]
    domain_top2_proba = domain_top3[1][1] if len(domain_top3) > 1 else 0.0
    priority_top1_label, priority_top1_proba = priority_top3[0]

    confidence = min(domain_top1_proba, priority_top1_proba)
    domain_margin = domain_top1_proba - domain_top2_proba

    needs_human = (
        language == "other"
        or confidence < CONFIDENCE_ABSTAIN_THRESHOLD
        or domain_margin < DOMAIN_MARGIN_THRESHOLD
    )

    domain = OTHER_DOMAIN if needs_human else domain_top1_label
    priority = priority_top1_label
    department = get_department(domain)
    sla_hours = get_sla_hours(priority)

    return {
        "language": language,
        "domain": domain,
        "department": department,
        "priority": priority,
        "confidence": round(float(confidence), 4),
        "needs_human": bool(needs_human),
        "sla_hours": sla_hours,
        "domain_top3": domain_top3,
        "priority_top3": priority_top3,
        "channel": channel,
        "clean_text": clean_text,
    }
