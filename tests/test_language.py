"""
Language detection tests (no API key required).

Cover:
- Pure English complaint -> "en"
- Pure Hindi (Devanagari) complaint -> "hi"
- Romanized Hinglish civic terms (bijli, naali, kachra) -> "hinglish"
- Tamil-only / random glyphs -> "other"
"""
from nagrikmitra.language import detect_language


def test_pure_english():
    text = "There is a huge pothole on the main road near my house, please fix it soon."
    assert detect_language(text) == "en"


def test_pure_hindi_devanagari():
    text = "मेरे घर के पास सड़क में बहुत बड़ा गड्ढा है, कृपया ठीक करें।"
    assert detect_language(text) == "hi"


def test_hinglish_bijli():
    text = "bijli transformer se sparking ho rahi hai, khatarnak hai"
    assert detect_language(text) == "hinglish"


def test_hinglish_naali():
    text = "naali me pani jama ho gaya hai, badbu aa rahi hai"
    assert detect_language(text) == "hinglish"


def test_hinglish_kachra():
    text = "kachra 3 din se nahi utha, bahut samasya ho rahi hai"
    assert detect_language(text) == "hinglish"


def test_tamil_only_is_other():
    text = "இது ஒரு தமிழ் வாக்கியம், சாலையில் பள்ளம் உள்ளது."
    assert detect_language(text) == "other"


def test_random_glyphs_is_other():
    text = "░▒▓■□▲▼★☆♠"
    assert detect_language(text) == "other"


def test_empty_string_is_other():
    assert detect_language("") == "other"
