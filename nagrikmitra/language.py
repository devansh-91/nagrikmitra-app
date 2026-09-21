"""
Language detection: hi / en / hinglish / other.

Responsibilities:
- Script detection (Devanagari unicode range vs Latin).
- Hinglish lexicon match (romanized Hindi civic words: bijli, naali, kachra, jal, ...).
- Unknown/non-Indic scripts (e.g. Tamil, random glyphs) -> "other".
- detect_language(text: str) -> Literal["hi", "en", "hinglish", "other"]
"""
import re
from typing import Literal

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Romanized Hindi civic-complaint vocabulary. Enough hits on Latin script
# text flips the detected language from "en" to "hinglish".
# Romanized Hindi words only — deliberately excludes English civic nouns
# (e.g. "streetlight", "pipeline", "complaint") that would otherwise false-
# -positive pure-English sentences mentioning civic infrastructure.
HINGLISH_LEXICON = {
    "bijli", "bijlee", "bijali", "naali", "nali", "nala", "kachra", "kachara",
    "jal", "paani", "pani", "sadak", "sadhak", "gaddha", "gadha", "gadde",
    "toota", "tuta", "tootu", "khraab", "kharab", "kharaab", "shikayat",
    "samasya", "bahut", "hai", "hain", "nahi", "nahin", "raha", "rahi",
    "rahe", "kaam", "kam", "safai", "safaai", "gandagi", "ganda", "machar",
    "machhar", "bimari", "beemari", "khatarnak", "khtarnak", "gali",
    "mohalla", "mohallay", "nagar", "nigam", "adhikari", "roz", "hafte",
    "hafto", "din", "dino", "mahine", "se", "ho", "wala", "wali", "jaldi",
    "turant", "khambha", "khamba", "batti", "seepage", "kooda", "kuda",
    "aspatal", "ghante", "ghanta", "band", "bhar", "gaya", "gayi", "aur",
    "kya", "kaise", "kab", "kahan", "karo", "karwao", "karwai", "dena",
    "milne", "mamuli", "zaroori", "zyada", "turnat", "thoda", "thodi",
    "accha", "theek", "karke", "jama",
}

TAMIL_RE = re.compile(r"[஀-௿]")


def _tokenize(text: str) -> list:
    return re.findall(r"[A-Za-z]+", text.lower())


def detect_language(text: str) -> Literal["hi", "en", "hinglish", "other"]:
    """Detect the dominant language/script of a civic complaint."""
    if not text or not text.strip():
        return "other"

    has_devanagari = bool(DEVANAGARI_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))

    if has_devanagari:
        return "hi"

    if not has_latin:
        # Non-Devanagari, non-Latin scripts (Tamil, random glyphs, emoji-only, ...).
        return "other"

    tokens = _tokenize(text)
    if not tokens:
        return "other"

    hinglish_hits = sum(1 for t in tokens if t in HINGLISH_LEXICON)
    if hinglish_hits >= 1:
        return "hinglish"

    return "en"
