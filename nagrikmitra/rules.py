"""
Keyword/rules baseline classifier — written from scratch, no ML.

Responsibilities:
- Simple keyword -> domain dictionary covering all 8 domains (en + hi/hinglish terms).
- predict_domain_keyword(text: str) -> str
- Used ONLY as a baseline to compare against the trained sklearn model in
  reports/metrics.json. Never presented as the production classifier.
"""
import re

from nagrikmitra.taxonomy import OTHER_DOMAIN

# Domain -> keyword list (English + romanized Hindi/Hinglish + Devanagari).
DOMAIN_KEYWORDS = {
    "Roads": [
        "pothole", "potholes", "road", "roads", "street", "footpath",
        "pavement", "crack", "divider", "speed breaker", "flyover",
        "sadak", "gaddha", "gadha", "gadde", "sadhak", "tuta hua",
        "सड़क", "गड्ढा", "गड्ढे", "फुटपाथ",
    ],
    "Lights": [
        "streetlight", "street light", "streetlamp", "street lamp",
        "lamp post", "pole light", "bulb", "khambha", "khamba", "batti",
        "streetlight khambha", "dark street", "स्ट्रीट लाइट", "बत्ती",
        "खंभा", "light not working", "no light",
    ],
    "Water": [
        "water", "pipeline", "pipe leak", "water leak", "water supply",
        "tap", "borewell", "tanker", "jal", "paani", "pani", "pipeline leak",
        "जल", "पानी", "नल", "पाइपलाइन",
    ],
    "Waste": [
        "garbage", "trash", "waste", "kachra", "kachara", "kooda", "kuda",
        "dustbin", "collection", "not collected", "कचरा", "कूड़ा",
        "डस्टबिन",
    ],
    "Sanitation": [
        "sewer", "sewage", "drain", "drainage", "naali", "nali", "nala",
        "overflow", "gutter", "manhole", "नाली", "सीवर", "गटर",
    ],
    "Electricity": [
        "electricity", "power cut", "transformer", "sparking", "short circuit",
        "voltage", "outage", "bijli", "bijlee", "bijali", "बिजली",
        "ट्रांसफार्मर", "power outage", "no power",
    ],
    "Health": [
        "dengue", "malaria", "mosquito", "machhar", "machar", "epidemic",
        "hospital", "clinic", "disease", "outbreak", "bimari", "beemari",
        "अस्पताल", "बीमारी", "मच्छर", "डेंगू",
    ],
}


def predict_domain_keyword(text: str) -> str:
    """Predict a domain using simple keyword matching. Falls back to
    'Other / Unknown' if no keyword hits any domain."""
    if not text:
        return OTHER_DOMAIN

    lowered = text.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            kw_lower = kw.lower()
            if re.search(r"[ऀ-ॿ]", kw_lower):
                # Devanagari keywords: plain substring match.
                if kw_lower in lowered:
                    scores[domain] += 1
            else:
                # Latin keywords: word-boundary match to avoid partial hits.
                if re.search(r"\b" + re.escape(kw_lower) + r"\b", lowered):
                    scores[domain] += 1

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return OTHER_DOMAIN
    return best_domain
