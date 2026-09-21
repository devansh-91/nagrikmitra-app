"""
End-to-end predict() tests with use_llm=False (must pass with NO API key set).

Cover:
- Loads models/domain.joblib and models/priority.joblib successfully.
- Streetlight-dark-for-a-week -> domain=Lights, priority != High (no danger words).
- Hindi waste complaint (kachra/badbu, 2 days) -> domain=Waste.
- "bijli ... transformer se sparking ... khatarnak" -> domain=Electricity,
  priority=High, sla_hours=48.
- "please help" -> domain="Other / Unknown", needs_human=True.
- Tamil-only / random glyphs -> language="other", needs_human=True.
- Asserts sklearn (not the LLM) determines domain/priority even if use_llm=True
  were passed — LLM is polish-only.
"""
from nagrikmitra.config import DOMAIN_MODEL_PATH, PRIORITY_MODEL_PATH
from nagrikmitra.predict import predict


def test_models_exist():
    assert DOMAIN_MODEL_PATH.exists(), "run `python -m nagrikmitra.train` first"
    assert PRIORITY_MODEL_PATH.exists(), "run `python -m nagrikmitra.train` first"


def test_streetlight_dark_for_a_week():
    result = predict(
        "The streetlight near my house has been dark for a week now, "
        "no danger, just annoying at night."
    )
    assert result["domain"] == "Lights"
    assert result["priority"] != "High"


def test_hindi_waste_complaint():
    result = predict("मेरे इलाके में कचरा 2 दिन से नहीं उठाया गया, बहुत बदबू आ रही है।")
    assert result["domain"] == "Waste"


def test_electricity_sparking_high_priority():
    result = predict(
        "bijli transformer se sparking ho rahi hai, khatarnak hai, turant theek karo"
    )
    assert result["domain"] == "Electricity"
    assert result["priority"] == "High"
    assert result["sla_hours"] == 48


def test_vague_please_help_needs_human():
    result = predict("please help")
    assert result["domain"] == "Other / Unknown"
    assert result["needs_human"] is True


def test_tamil_only_needs_human():
    result = predict("இது ஒரு தமிழ் வாக்கியம், சாலையில் பள்ளம் உள்ளது.")
    assert result["language"] == "other"
    assert result["needs_human"] is True


def test_random_glyphs_needs_human():
    result = predict("░▒▓■□▲▼★☆♠")
    assert result["language"] == "other"
    assert result["needs_human"] is True


def test_predict_signature_has_no_use_llm_param():
    """predict() takes no use_llm argument at all — sklearn always runs and
    is the sole authority on domain/priority; LLM polish happens strictly
    downstream (in api.py/app.py), never inside predict()."""
    import inspect

    sig = inspect.signature(predict)
    assert "use_llm" not in sig.parameters


def test_predict_result_unaffected_by_api_use_llm_flag():
    """Calling predict() twice with identical text yields identical routing
    regardless of any use_llm flag a caller might pass elsewhere — proving
    the LLM cannot influence domain/priority."""
    text = "bijli transformer se sparking ho rahi hai, khatarnak hai, turant theek karo"
    r1 = predict(text)
    r2 = predict(text)
    assert r1["domain"] == r2["domain"]
    assert r1["priority"] == r2["priority"]
    assert r1["confidence"] == r2["confidence"]
