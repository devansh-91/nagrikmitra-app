"""
Template acknowledgment replies (per detected language) with a locked SLA number.

Responsibilities:
- render_template(language, domain, priority, sla_hours) -> str
- Templates for hi / en / hinglish (and a neutral fallback for "other").
- sla_regex(sla_hours) -> compiled regex requiring the number to be
  attached to a unit: hour(s) / hrs / h / ghante, e.g.
  (?<!\\d){sla}\\s*(hours?|hrs?|h|ghante)
  (guards against false positives like "P1" or "24 hours" matching on "1"/"4").
"""
import re

_TEMPLATES = {
    "en": (
        "Thank you for reporting this {domain} issue. It has been logged "
        "with {department} at {priority} priority. Expected response time: "
        "{sla_hours} hours. We will keep you updated."
    ),
    "hi": (
        "आपकी {domain} संबंधी शिकायत के लिए धन्यवाद। इसे {department} के पास "
        "{priority} प्राथमिकता के साथ दर्ज कर लिया गया है। अपेक्षित प्रतिक्रिया समय: "
        "{sla_hours} ghante. हम आपको अपडेट करते रहेंगे।"
    ),
    "hinglish": (
        "Aapki {domain} complaint ke liye dhanyavaad. Ise {department} ke paas "
        "{priority} priority ke saath darj kar liya gaya hai. Expected response "
        "time: {sla_hours} ghante. Hum aapko update karte rahenge."
    ),
    "other": (
        "Thank you for your complaint. It has been logged with {department} "
        "at {priority} priority for human review. Expected response time: "
        "{sla_hours} hours. We will keep you updated."
    ),
}


def render_template(language: str, domain: str, priority: str, sla_hours: int, department: str = "") -> str:
    """Render the ticket-language acknowledgment with a locked SLA number."""
    template = _TEMPLATES.get(language, _TEMPLATES["other"])
    return template.format(
        domain=domain,
        department=department or "the relevant department",
        priority=priority,
        sla_hours=sla_hours,
    )


def sla_regex(sla_hours: int) -> re.Pattern:
    """Compiled regex requiring the SLA number to be attached to a time
    unit (hour(s)/hrs/h/ghante), guarding against substring false positives
    like "P1" or "148 hours" matching a bare "1" or "4"."""
    return re.compile(
        rf"(?<!\d){sla_hours}\s*(hours?|hrs?|h|ghante|ghanta)\b",
        re.IGNORECASE,
    )
