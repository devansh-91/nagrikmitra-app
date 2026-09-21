"""
Optional LLM polish of the TEMPLATE reply only. Never touches routing.

Responsibilities:
- Groq client (reads GROQ_API_KEY / GROQ_MODEL from env); on first real key,
  list available models and pick one that actually exists (do not hardcode
  llama-3.1-8b-instant — this account has used openai/gpt-oss-20b).
- Ollama local fallback if no Groq key (OLLAMA_HOST / OLLAMA_MODEL).
- polish(template_reply, sla_hours, language) -> (text, used: bool, backend, error)
- MUST validate the SLA number survives (via reply.sla_regex) before accepting
  the LLM output. On 400 / timeout / dropped SLA number -> fall back to the
  template unmodified, used=False, llm_error set.
"""
import requests

from nagrikmitra.config import GROQ_API_KEY, GROQ_MODEL, OLLAMA_HOST, OLLAMA_MODEL
from nagrikmitra.reply import sla_regex

_LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "hinglish": "Hinglish (romanized Hindi mixed with English)",
    "other": "the same language as the input",
}

_SYSTEM_PROMPT = (
    "You are polishing a short civic-complaint acknowledgment message. "
    "Rewrite the given message to sound warmer and more natural in {language}, "
    "while keeping it concise (2-3 sentences). "
    "You MUST preserve the exact SLA number and its time unit "
    "(e.g. '{sla_hours} hours' or '{sla_hours} ghante') verbatim, unchanged. "
    "Do not add any new commitments, promises, or information. "
    "Output ONLY the rewritten message, nothing else."
)


def _validate(text: str, sla_hours: int) -> bool:
    return bool(text) and bool(sla_regex(sla_hours).search(text))


def _groq_available_models() -> list:
    """List models on the Groq account (used to pick a real model on first
    setup rather than hardcoding one that may not exist)."""
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception:
        return []


def _polish_groq(template_reply: str, sla_hours: int, language: str):
    try:
        from groq import Groq
    except ImportError:
        return None, False, "groq", "groq package not installed"

    try:
        client = Groq(api_key=GROQ_API_KEY)
        lang_name = _LANGUAGE_NAMES.get(language, "the same language as the input")
        system_prompt = _SYSTEM_PROMPT.format(language=lang_name, sla_hours=sla_hours)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": template_reply},
            ],
            temperature=0.3,
            max_tokens=300,
            timeout=15,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as exc:
        return None, False, "groq", f"groq error: {exc}"

    if not _validate(text, sla_hours):
        return None, False, "groq", "SLA number missing/altered in LLM output"

    return text, True, "groq", None


def _polish_ollama(template_reply: str, sla_hours: int, language: str):
    lang_name = _LANGUAGE_NAMES.get(language, "the same language as the input")
    system_prompt = _SYSTEM_PROMPT.format(language=lang_name, sla_hours=sla_hours)
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": template_reply},
                ],
                "stream": False,
            },
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
    except Exception as exc:
        return None, False, "ollama", f"ollama error: {exc}"

    if not _validate(text, sla_hours):
        return None, False, "ollama", "SLA number missing/altered in LLM output"

    return text, True, "ollama", None


def active_backend() -> str:
    """Which backend would be used: 'groq', 'ollama', or 'none'."""
    if GROQ_API_KEY:
        return "groq"
    return "ollama"


def polish(template_reply: str, sla_hours: int, language: str):
    """Polish the template reply via Groq (if a key is set) or Ollama
    (local fallback). Falls back to the unmodified template, used=False,
    on any failure or if the SLA number doesn't survive."""
    if GROQ_API_KEY:
        text, used, backend, error = _polish_groq(template_reply, sla_hours, language)
        if used:
            return text, used, backend, error
        # Fall through to template on Groq failure — do not silently try
        # Ollama, since the operator explicitly configured Groq.
        return template_reply, False, backend, error

    text, used, backend, error = _polish_ollama(template_reply, sla_hours, language)
    if used:
        return text, used, backend, error
    return template_reply, False, backend, error
