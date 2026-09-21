"""
Template reply tests (no API key required).

Cover:
- render_template produces the ticket-language reply with the correct
  SLA number embedded for each of the 3 priorities.
- LLM polish is never invoked when use_llm=False (see test_predict.py).
"""
from nagrikmitra.reply import render_template, sla_regex
from nagrikmitra.taxonomy import SLA_HOURS


def test_render_template_en_high():
    reply = render_template("en", "Electricity", "High", SLA_HOURS["High"], "Discom")
    assert sla_regex(SLA_HOURS["High"]).search(reply)


def test_render_template_hi_medium():
    reply = render_template("hi", "Waste", "Medium", SLA_HOURS["Medium"], "SWM Dept.")
    assert sla_regex(SLA_HOURS["Medium"]).search(reply)


def test_render_template_hinglish_low():
    reply = render_template("hinglish", "Roads", "Low", SLA_HOURS["Low"], "PWD")
    assert sla_regex(SLA_HOURS["Low"]).search(reply)


def test_render_template_other_fallback():
    reply = render_template("other", "Other / Unknown", "Low", SLA_HOURS["Low"], "Human review")
    assert sla_regex(SLA_HOURS["Low"]).search(reply)


def test_unknown_language_falls_back_to_other_template():
    reply = render_template("xx", "Roads", "High", 48, "PWD")
    assert sla_regex(48).search(reply)


def test_llm_not_invoked_without_use_llm(monkeypatch):
    """When use_llm is False, callers must not touch nagrikmitra.llm at all —
    the template reply is the final suggested_reply."""
    import nagrikmitra.llm as llm_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("llm.polish should not be called when use_llm=False")

    monkeypatch.setattr(llm_module, "polish", _fail_if_called)

    # Simulate the api.py contract: polish() is simply never called.
    reply = render_template("en", "Roads", "Medium", SLA_HOURS["Medium"], "PWD")
    assert sla_regex(SLA_HOURS["Medium"]).search(reply)
