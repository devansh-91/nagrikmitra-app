"""
Text normalization shared by training and inference.

Responsibilities:
- Unicode NFC normalization.
- Strip URLs, excessive punctuation/noise ("!!!", "pls", "urgent") without
  destroying signal used by priority head.
- Preserve Devanagari + Latin characters.
- normalize(text: str) -> str
"""
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
# Collapse runs of 3+ repeated punctuation (e.g. "!!!", "???", "...") to a single char.
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
# Keep Devanagari, Latin letters/digits, and common punctuation; drop stray symbols.
ALLOWED_CHARS_RE = re.compile(
    r"[^ऀ-ॿA-Za-z0-9\s.,!?/:;()\-'\"@#]"
)


def normalize(text: str) -> str:
    """Normalize complaint text for training/inference: NFC, strip URLs/noise,
    preserve Devanagari + Latin signal."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = ALLOWED_CHARS_RE.sub(" ", text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text
