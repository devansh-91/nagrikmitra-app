"""
Synthetic data generator -> data/complaints.csv (~1200 rows, seed=42).

Responsibilities:
- 8 domains (incl. Other/Unknown), 3 priorities, en/hi/hinglish languages.
- Realistic civic complaint templates (potholes, bijli sparking, naali overflow,
  kachra, dengue, streetlight outages, jal leaks, etc.) with light noise.
- Channels: web, app, whatsapp, phone.
- Columns: id, text, domain, priority, language, channel, timestamp.
- Deterministic via random.seed(42) / np.random.seed(42).

Run: python -m nagrikmitra.generate_data
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from nagrikmitra.config import COMPLAINTS_CSV_PATH

SEED = 42

LOCATIONS = [
    "Sector 21", "MG Road", "Lajpat Nagar", "Rohini", "Andheri West",
    "Koramangala", "Salt Lake", "Civil Lines", "Indira Nagar", "Shastri Nagar",
    "Model Town", "Saket", "Dwarka Sector 9", "Vasant Kunj", "Malviya Nagar",
    "Gandhi Chowk", "Nehru Colony", "Ashok Vihar", "Hauz Khas", "Patel Nagar",
    "Ward 12", "Ward 7", "Green Park", "Kalkaji", "Mayur Vihar",
]

DURATIONS = {
    "en": ["3 days", "a week", "10 days", "2 weeks", "a month", "5 days"],
    "hi": ["3 दिन", "एक हफ्ते", "10 दिनों", "2 हफ्तों", "एक महीने", "5 दिनों"],
    "hinglish": ["3 din", "ek hafte", "10 dino", "2 hafto", "ek mahine", "5 din"],
}

CHANNELS = ["web", "app", "whatsapp", "phone"]

# Cross-priority vocabulary blended in at low frequency so the priority head
# can't perfectly separate classes on keyword presence alone (real
# complainants exaggerate or understate urgency in their own words).
PRIORITY_BLEND_WORDS = {
    "en": {
        "High": ["urgent", "dangerous", "immediately", "emergency"],
        "Medium": ["please look into this", "getting worse", "need this fixed"],
        "Low": ["suggestion", "not urgent", "whenever convenient"],
    },
    "hi": {
        "High": ["तुरंत", "खतरनाक", "आपातकाल"],
        "Medium": ["कृपया ध्यान दें", "हालत बिगड़ रही है"],
        "Low": ["सुझाव", "जरूरी नहीं", "जब समय मिले"],
    },
    "hinglish": {
        "High": ["turant", "khatarnak", "emergency"],
        "Medium": ["please dhyan do", "halat bigad rahi hai"],
        "Low": ["suggestion", "urgent nahi", "jab time mile"],
    },
}
_ADJACENT_PRIORITY = {"High": "Medium", "Medium": "Low", "Low": "Medium"}

NOISE_SUFFIXES = {
    "en": ["", "", " pls fix asap", " urgent!!!", " please help", ""],
    "hi": ["", "", " कृपया जल्दी करें", "", " बहुत परेशानी हो रही है", ""],
    "hinglish": ["", "", " pls jaldi karo", " urgent hai", "", ""],
}

# --- Templates: domain -> language -> priority -> [template strings] ---
# {loc} and {dur} are filled from LOCATIONS / DURATIONS[lang].
TEMPLATES = {
    "Roads": {
        "en": {
            "High": [
                "There is a dangerous pothole near {loc} that caused an accident yesterday, please fix urgently.",
                "A huge crater has opened up on the road at {loc}, two bikes have already skidded, extremely dangerous.",
            ],
            "Medium": [
                "There is a large pothole on the road near {loc} causing traffic jams for {dur}.",
                "The road near {loc} has multiple potholes making it hard to drive, been like this for {dur}.",
            ],
            "Low": [
                "Minor crack developing on the footpath near {loc}, suggest repairing before monsoon.",
                "Just a small pothole near {loc}, not urgent but should be patched sometime.",
            ],
        },
        "hi": {
            "High": [
                "{loc} के पास सड़क में बहुत बड़ा गड्ढा है जिसकी वजह से कल एक दुर्घटना हो गई, कृपया तुरंत ठीक करें।",
                "{loc} में सड़क अचानक धंस गई है, यह बहुत खतरनाक है, जल्द मरम्मत करें।",
            ],
            "Medium": [
                "{loc} के पास सड़क में {dur} से गड्ढा है जिससे ट्रैफिक जाम हो रहा है।",
                "{loc} में सड़क की हालत {dur} से खराब है, कृपया मरम्मत करवाएं।",
            ],
            "Low": [
                "{loc} के पास फुटपाथ में छोटी सी दरार है, समय मिलने पर ठीक करवा दें।",
                "सुझाव: {loc} की सड़क का पुनर्निर्माण कर दिया जाए तो अच्छा रहेगा।",
            ],
        },
        "hinglish": {
            "High": [
                "{loc} ke paas sadak me bahut bada gaddha hai jiski wajah se kal accident ho gaya, please turant fix karo.",
                "{loc} me sadak achanak dhans gayi hai, bahut khatarnak hai, jaldi repair karwao.",
            ],
            "Medium": [
                "{loc} ke paas sadak me {dur} se gaddha hai jisse traffic jam ho raha hai.",
                "{loc} me sadak ki halat {dur} se kharab hai, please theek karwao.",
            ],
            "Low": [
                "{loc} ke paas footpath me chhoti si darar hai, time milne par theek karwa dena.",
                "Suggestion: {loc} ki sadak resurface kar di jaye to accha rahega.",
            ],
        },
    },
    "Lights": {
        "en": {
            "High": [
                "Streetlight near {loc} has been dark for {dur} and there was a mugging last night, women feel unsafe, urgent action needed.",
                "All streetlights on the main stretch near {loc} are out, pitch dark at night, very dangerous for commuters.",
            ],
            "Medium": [
                "Streetlight near {loc} has been flickering and going off for {dur} now.",
                "The lamp post near {loc} has been dark for {dur}, please send a technician.",
            ],
            "Low": [
                "One streetlight bulb near {loc} is dim, not urgent but could be replaced.",
                "Suggestion: add an extra streetlight near {loc} park for better visibility.",
            ],
        },
        "hi": {
            "High": [
                "{loc} के पास स्ट्रीट लाइट {dur} से बंद है और कल रात लूट हो गई, महिलाएं असुरक्षित महसूस कर रही हैं, तुरंत ठीक करें।",
                "{loc} की सभी स्ट्रीट लाइटें बंद हैं, रात में पूरा अंधेरा रहता है, बहुत खतरनाक है।",
            ],
            "Medium": [
                "{loc} के पास स्ट्रीट लाइट {dur} से टिमटिमा रही है और बंद हो जाती है।",
                "{loc} में खंभे की बत्ती {dur} से बंद है, कृपया तकनीशियन भेजें।",
            ],
            "Low": [
                "{loc} के पास एक स्ट्रीट लाइट थोड़ी धीमी है, जरूरी नहीं पर बदल दी जाए तो अच्छा है।",
                "सुझाव: {loc} के पार्क में एक और स्ट्रीट लाइट लगाई जाए।",
            ],
        },
        "hinglish": {
            "High": [
                "{loc} ke paas streetlight {dur} se band hai aur kal raat loot ho gayi, mahilaye unsafe feel kar rahi hai, turant theek karo.",
                "{loc} ki sabhi streetlights band hai, raat me pura andhera rehta hai, bahut khatarnak hai.",
            ],
            "Medium": [
                "{loc} ke paas streetlight {dur} se flicker kar rahi hai aur band ho jati hai.",
                "{loc} me khambe ki batti {dur} se band hai, please technician bhejo.",
            ],
            "Low": [
                "{loc} ke paas ek streetlight thodi dim hai, urgent nahi par badal di jaye to accha hai.",
                "Suggestion: {loc} ke park me ek aur streetlight lagayi jaye.",
            ],
        },
    },
    "Water": {
        "en": {
            "High": [
                "A major water pipeline near {loc} has burst and is flooding homes, contaminated water mixing in, urgent!",
                "Sewage is mixing into the drinking water supply near {loc}, families are falling sick, immediate action needed.",
            ],
            "Medium": [
                "Water supply near {loc} has been irregular for {dur}, taps run dry most mornings.",
                "Low water pressure near {loc} for {dur} now, tankers needed in the meantime.",
            ],
            "Low": [
                "Minor drip from a water pipe near {loc}, not urgent but should be checked.",
                "Suggestion: increase water supply timing near {loc} by half an hour.",
            ],
        },
        "hi": {
            "High": [
                "{loc} के पास पानी की मुख्य पाइपलाइन फट गई है और घरों में पानी भर रहा है, गंदा पानी मिल रहा है, तुरंत कार्रवाई करें।",
                "{loc} के पास पीने के पानी में सीवेज मिल रहा है, परिवार बीमार पड़ रहे हैं, तत्काल ध्यान दें।",
            ],
            "Medium": [
                "{loc} में पानी की सप्लाई {dur} से अनियमित है, सुबह नल सूखे रहते हैं।",
                "{loc} के पास {dur} से पानी का प्रेशर बहुत कम है, टैंकर की जरूरत है।",
            ],
            "Low": [
                "{loc} के पास पानी के पाइप से हल्का रिसाव है, जरूरी नहीं पर जांच करवा लें।",
                "सुझाव: {loc} में पानी सप्लाई का समय आधा घंटा बढ़ा दिया जाए।",
            ],
        },
        "hinglish": {
            "High": [
                "{loc} ke paas pani ki main pipeline fat gayi hai aur gharo me pani bhar raha hai, ganda jal mil raha hai, turant karwai karo.",
                "{loc} ke paas peene ke paani me sewage mix ho raha hai, family bimar pad rahi hai, turant dhyan do.",
            ],
            "Medium": [
                "{loc} me paani ki supply {dur} se irregular hai, subah nal sookhe rehte hai.",
                "{loc} ke paas {dur} se paani ka pressure bahut kam hai, tanker ki zarurat hai.",
            ],
            "Low": [
                "{loc} ke paas paani ke pipe se halka leak hai, urgent nahi par check karwa lena.",
                "Suggestion: {loc} me paani supply ka time aadha ghanta badha diya jaye.",
            ],
        },
    },
    "Waste": {
        "en": {
            "High": [
                "Garbage has piled up near {loc} for {dur}, medical waste is mixed in, rats and stray dogs are swarming, health hazard.",
                "A huge trash pile near {loc} is rotting and attracting disease, urgent collection needed before it spreads illness.",
            ],
            "Medium": [
                "Garbage has not been collected near {loc} for {dur} now, bins are overflowing.",
                "Waste collection near {loc} has been delayed for {dur}, smell is getting bad.",
            ],
            "Low": [
                "Suggestion: please increase garbage collection frequency near {loc} to twice a week.",
                "Minor litter accumulating near {loc}, not urgent but worth a look.",
            ],
        },
        "hi": {
            "High": [
                "{loc} के पास {dur} से कचरे का ढेर लगा है, मेडिकल वेस्ट भी मिला है, चूहे और आवारा कुत्ते बढ़ रहे हैं, स्वास्थ्य के लिए खतरा है।",
                "{loc} के पास कचरे का बड़ा ढेर सड़ रहा है और बीमारी फैला रहा है, तुरंत उठवाएं।",
            ],
            "Medium": [
                "{loc} के पास {dur} से कचरा नहीं उठाया गया है, डस्टबिन भर गए हैं।",
                "{loc} में कचरा उठाने में {dur} से देरी हो रही है, बदबू बढ़ रही है।",
            ],
            "Low": [
                "सुझाव: {loc} में कचरा उठाने की संख्या हफ्ते में दो बार कर दी जाए।",
                "{loc} के पास हल्का कचरा जमा हो रहा है, जरूरी नहीं पर देख लें।",
            ],
        },
        "hinglish": {
            "High": [
                "{loc} ke paas {dur} se kachre ka dher laga hai, medical waste bhi mila hai, chuhe aur awara kutte badh rahe hai, health hazard hai.",
                "{loc} ke paas kachre ka bada dher sad raha hai aur bimari fela raha hai, turant uthwao.",
            ],
            "Medium": [
                "{loc} ke paas {dur} se kachra nahi uthaya gaya hai, dustbin bhar gaye hai.",
                "{loc} me kachra uthane me {dur} se deri ho rahi hai, badbu badh rahi hai.",
            ],
            "Low": [
                "Suggestion: {loc} me kachra uthane ki frequency hafte me do baar kar di jaye.",
                "{loc} ke paas halka kachra jama ho raha hai, urgent nahi par dekh lena.",
            ],
        },
    },
    "Sanitation": {
        "en": {
            "High": [
                "Sewage is overflowing into homes near {loc}, extremely unhygienic and dangerous for children, urgent help needed.",
                "The drain near {loc} has burst and sewage water is flooding the street, health emergency.",
            ],
            "Medium": [
                "The drain near {loc} has been blocked for {dur}, water is stagnating.",
                "Naali near {loc} is clogged for {dur}, foul smell spreading in the area.",
            ],
            "Low": [
                "Minor smell from the drain near {loc}, not urgent but could be cleaned.",
                "Suggestion: cover the open drain near {loc} for safety.",
            ],
        },
        "hi": {
            "High": [
                "{loc} के पास सीवेज घरों में घुस रहा है, बच्चों के लिए बहुत खतरनाक और गंदा है, तुरंत मदद चाहिए।",
                "{loc} के पास नाली फट गई है और सीवेज का पानी सड़क पर भर गया है, स्वास्थ्य आपातकाल है।",
            ],
            "Medium": [
                "{loc} के पास नाली {dur} से बंद है, पानी जमा हो रहा है।",
                "{loc} में नाली {dur} से जाम है, बदबू फैल रही है।",
            ],
            "Low": [
                "{loc} के पास नाली से हल्की बदबू आ रही है, जरूरी नहीं पर सफाई करवा दें।",
                "सुझाव: {loc} की खुली नाली को सुरक्षा के लिए ढक दिया जाए।",
            ],
        },
        "hinglish": {
            "High": [
                "{loc} ke paas sewage gharo me ghus raha hai, bacho ke liye bahut khatarnak aur ganda hai, turant madad chahiye.",
                "{loc} ke paas naali fat gayi hai aur sewage ka pani sadak par bhar gaya hai, health emergency hai.",
            ],
            "Medium": [
                "{loc} ke paas naali {dur} se band hai, pani jama ho raha hai.",
                "{loc} me naali {dur} se jam hai, badbu fail rahi hai.",
            ],
            "Low": [
                "{loc} ke paas naali se halki badbu aa rahi hai, urgent nahi par safai karwa dena.",
                "Suggestion: {loc} ki khuli naali ko safety ke liye dhak diya jaye.",
            ],
        },
    },
    "Electricity": {
        "en": {
            "High": [
                "The transformer near {loc} is sparking badly, extremely khatarnak, could catch fire, urgent action needed.",
                "A live wire has fallen near {loc} after last night's storm, very dangerous, someone could get electrocuted.",
            ],
            "Medium": [
                "Power keeps fluctuating near {loc} for {dur}, damaging appliances.",
                "There have been frequent power cuts near {loc} for {dur} now.",
            ],
            "Low": [
                "Minor voltage flicker near {loc}, not urgent but worth checking.",
                "Suggestion: upgrade the old wiring near {loc} sometime this year.",
            ],
        },
        "hi": {
            "High": [
                "{loc} के पास ट्रांसफार्मर से चिंगारी निकल रही है, बहुत खतरनाक है, आग लग सकती है, तुरंत कार्रवाई करें।",
                "{loc} के पास कल रात तूफान के बाद तार टूटकर गिर गया है, बहुत खतरनाक है, किसी को करंट लग सकता है।",
            ],
            "Medium": [
                "{loc} के पास {dur} से बिजली बार-बार जा रही है, उपकरण खराब हो रहे हैं।",
                "{loc} में {dur} से बार-बार बिजली कटौती हो रही है।",
            ],
            "Low": [
                "{loc} के पास हल्का वोल्टेज उतार-चढ़ाव है, जरूरी नहीं पर जांच करवा लें।",
                "सुझाव: {loc} की पुरानी वायरिंग इस साल बदल दी जाए।",
            ],
        },
        "hinglish": {
            "High": [
                "{loc} ke paas transformer se sparking ho rahi hai, bahut khatarnak hai, aag lag sakti hai, turant karwai karo.",
                "{loc} ke paas kal raat tufan ke baad tar toot kar gir gaya hai, bahut khatarnak hai, kisi ko current lag sakta hai.",
            ],
            "Medium": [
                "{loc} ke paas {dur} se bijli baar baar ja rahi hai, appliances kharab ho rahe hai.",
                "{loc} me {dur} se baar baar power cut ho raha hai.",
            ],
            "Low": [
                "{loc} ke paas halka voltage fluctuation hai, urgent nahi par check karwa lena.",
                "Suggestion: {loc} ki purani wiring is saal badal di jaye.",
            ],
        },
    },
    "Health": {
        "en": {
            "High": [
                "Multiple dengue cases have been reported near {loc}, mosquito breeding is out of control, urgent fogging needed.",
                "There has been an accident near {loc} and no ambulance has arrived for over an hour, medical emergency.",
            ],
            "Medium": [
                "Mosquito breeding has increased near {loc} for {dur}, residents fear an outbreak.",
                "Stray dogs near {loc} have been aggressive for {dur}, a resident was bitten.",
            ],
            "Low": [
                "Suggestion: conduct a fogging drive near {loc} before monsoon season.",
                "Minor hygiene issue at the community clinic near {loc}, not urgent.",
            ],
        },
        "hi": {
            "High": [
                "{loc} के पास डेंगू के कई मामले सामने आए हैं, मच्छरों का प्रकोप बढ़ गया है, तुरंत फॉगिंग करवाएं।",
                "{loc} के पास एक्सीडेंट हुआ है और एक घंटे से एम्बुलेंस नहीं आई, चिकित्सा आपातकाल है।",
            ],
            "Medium": [
                "{loc} के पास {dur} से मच्छरों का प्रकोप बढ़ रहा है, लोग बीमारी फैलने से डर रहे हैं।",
                "{loc} के पास आवारा कुत्ते {dur} से आक्रामक हो गए हैं, एक व्यक्ति को काट लिया।",
            ],
            "Low": [
                "सुझाव: मानसून से पहले {loc} के पास फॉगिंग करवाई जाए।",
                "{loc} के पास कम्युनिटी क्लिनिक में मामूली सफाई की समस्या है, जरूरी नहीं।",
            ],
        },
        "hinglish": {
            "High": [
                "{loc} ke paas dengue ke kai cases aaye hai, machhar bahut badh gaye hai, turant fogging karwao.",
                "{loc} ke paas accident hua hai aur ek ghante se ambulance nahi aayi, medical emergency hai.",
            ],
            "Medium": [
                "{loc} ke paas {dur} se machhar badh rahe hai, log bimari failne se dar rahe hai.",
                "{loc} ke paas awara kutte {dur} se aggressive ho gaye hai, ek insaan ko kaat liya.",
            ],
            "Low": [
                "Suggestion: monsoon se pehle {loc} ke paas fogging karwai jaye.",
                "{loc} ke paas community clinic me mamuli safai ki problem hai, urgent nahi.",
            ],
        },
    },
    "Other / Unknown": {
        "en": {
            "High": [
                "A major fire hazard has been reported near {loc}, authorities must act immediately.",
                "Stray dog attacks near {loc} have injured two children, please act urgently.",
            ],
            "Medium": [
                "Illegal parking near {loc} has been blocking the road for {dur}, please take action.",
                "Loud construction noise near {loc} at night for {dur}, disturbing residents.",
            ],
            "Low": [
                "Suggestion: please add more benches in the park near {loc}.",
                "Minor encroachment issue near {loc}, not urgent, just flagging.",
            ],
        },
        "hi": {
            "High": [
                "{loc} के पास बड़ी आग लगने का खतरा बताया गया है, प्रशासन तुरंत कार्रवाई करे।",
                "{loc} के पास आवारा कुत्तों ने दो बच्चों को काट लिया, कृपया जल्द कार्रवाई करें।",
            ],
            "Medium": [
                "{loc} के पास {dur} से अवैध पार्किंग सड़क को जाम कर रही है, कृपया कार्रवाई करें।",
                "{loc} में रात को निर्माण कार्य का बहुत शोर हो रहा है, {dur} से यही हाल है।",
            ],
            "Low": [
                "सुझाव: {loc} के पार्क में और बेंच लगाई जाएं।",
                "{loc} में मामूली अतिक्रमण है, ज्यादा जरूरी नहीं, बस सूचित कर रहा हूं।",
            ],
        },
        "hinglish": {
            "High": [
                "{loc} ke paas bade aag lagne ka khatra bataya gaya hai, prashasan turant karwai kare.",
                "{loc} ke paas awara kutto ne do bachho ko kaat liya, please jaldi karwai karo.",
            ],
            "Medium": [
                "{loc} ke paas {dur} se illegal parking road ko jam kar rahi hai, please karwai karo.",
                "{loc} me raat ko construction ka bahut shor ho raha hai, {dur} se yahi haal hai.",
            ],
            "Low": [
                "Suggestion: {loc} ke park me aur benches lagayi jaye.",
                "{loc} me mamuli encroachment hai, zyada zaroori nahi, bas inform kar raha hu.",
            ],
        },
    },
}

DOMAINS = list(TEMPLATES.keys())
LANGUAGES = ["en", "hi", "hinglish"]
# Rows per domain, split by priority: High=45, Medium=60, Low=45 -> 150/domain, 1200 total.
PRIORITY_COUNTS = {"High": 45, "Medium": 60, "Low": 45}

ANCHOR_DATE = datetime(2025, 1, 1)


def _typo(word: str, rng: random.Random) -> str:
    """Apply one light character-level typo to a Latin word (swap/drop/dup)."""
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randint(1, len(word) - 2)
    kind = rng.choice(["swap", "drop", "dup"])
    if kind == "swap":
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if kind == "drop":
        return word[:i] + word[i + 1 :]
    return word[:i] + word[i] + word[i:]


def _add_noise(text: str, rng: random.Random) -> str:
    """Inject light realistic noise (typos, a dropped word) into a fraction
    of rows so the dataset isn't trivially memorized by exact template match."""
    words = text.split(" ")
    if not words:
        return text

    # ~35% chance: typo one random word.
    if rng.random() < 0.35 and len(words) > 2:
        idx = rng.randrange(len(words))
        words[idx] = _typo(words[idx], rng)

    # ~15% chance: drop one non-essential (short) word, e.g. filler.
    if rng.random() < 0.15 and len(words) > 5:
        candidates = [i for i, w in enumerate(words) if len(w) <= 3]
        if candidates:
            words.pop(rng.choice(candidates))

    return " ".join(words)


def _blend_vocab(text: str, priority: str, lang: str, rng: random.Random) -> str:
    """With ~18% probability, borrow one phrase from an adjacent priority's
    vocabulary so priority classes overlap lexically at the margins,
    mirroring how real complainants over/understate urgency."""
    if rng.random() >= 0.18:
        return text
    neighbor = _ADJACENT_PRIORITY[priority]
    phrase = rng.choice(PRIORITY_BLEND_WORDS[lang][neighbor])
    return f"{text} {phrase}"


def _fill(template: str, rng: random.Random, lang: str, priority: str) -> str:
    text = template.format(
        loc=rng.choice(LOCATIONS),
        dur=rng.choice(DURATIONS[lang]),
    )
    suffix = rng.choice(NOISE_SUFFIXES[lang])
    text = (text + suffix).strip()
    text = _blend_vocab(text, priority, lang, rng)
    return _add_noise(text, rng)


def generate(seed: int = SEED) -> pd.DataFrame:
    """Generate the synthetic complaints dataset deterministically."""
    rng = random.Random(seed)
    np.random.seed(seed)

    rows = []
    row_id = 1
    for domain in DOMAINS:
        for priority, total_count in PRIORITY_COUNTS.items():
            per_lang = total_count // len(LANGUAGES)
            for lang in LANGUAGES:
                templates = TEMPLATES[domain][lang][priority]
                for _ in range(per_lang):
                    template = rng.choice(templates)
                    text = _fill(template, rng, lang, priority)
                    channel = rng.choice(CHANNELS)
                    offset_days = rng.randint(0, 179)
                    offset_seconds = rng.randint(0, 86399)
                    timestamp = ANCHOR_DATE + timedelta(
                        days=offset_days, seconds=offset_seconds
                    )
                    rows.append(
                        {
                            "id": row_id,
                            "text": text,
                            "domain": domain,
                            "priority": priority,
                            "language": lang,
                            "channel": channel,
                            "timestamp": timestamp.isoformat(),
                        }
                    )
                    row_id += 1

    df = pd.DataFrame(rows)
    # Shuffle deterministically so the CSV isn't grouped by domain/priority.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)
    return df


def main() -> None:
    df = generate(SEED)
    COMPLAINTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPLAINTS_CSV_PATH, index=False)
    print(f"Wrote {len(df)} rows to {COMPLAINTS_CSV_PATH}")
    print(df["domain"].value_counts())
    print(df["priority"].value_counts())
    print(df["language"].value_counts())


if __name__ == "__main__":
    main()
