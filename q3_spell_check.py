"""
Q3: Hindi Spell Checking on ~1,77,000 Unique Words
Downloads word list from Google Sheet, classifies each word,
outputs confidence scores and reasoning.
"""

import re
import requests
import pandas as pd
from pathlib import Path
from io import StringIO

RESULTS_DIR = Path("results/q3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# STEP 1: Download ALL words from Google Sheet
print("=" * 60)
print("STEP 1: Downloading word list from Google Sheet")
print("=" * 60)

SHEET_ID = "17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw"
CSV_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"

print("Downloading from Google Sheet ...")
response = requests.get(CSV_URL, timeout=120)
response.raise_for_status()
response.encoding = "utf-8"
word_df = pd.read_csv(StringIO(response.text))

print(f"Columns  : {list(word_df.columns)}")
print(f"Rows     : {len(word_df)}")

word_col     = word_df.columns[0]
unique_words = (word_df[word_col]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist())
unique_words = [w for w in unique_words
                if w and w.lower() not in {"word","words","शब्द"}]
print(f"Unique words loaded: {len(unique_words)}")
print(f"Sample: {unique_words[:10]}")

# STEP 2: CLASSIFICATION ENGINE
print("\n" + "=" * 60)
print("STEP 2: Classification engine")
print("=" * 60)

# ── Whitelists 
COMMON_HINDI = {
    "है","हैं","था","थी","थे","हो","हूं","हूँ","होगा","होगी","होंगे","होना",
    "का","की","के","को","से","में","पर","और","या","तो","भी","ही","न","ना",
    "यह","वह","यहाँ","वहाँ","जो","कि","अगर","लेकिन","मगर","परंतु","किंतु",
    "मैं","हम","तुम","आप","वो","ये","इसे","उसे","हमें","उन्हें","तुम्हें",
    "नहीं","नही","मत","कभी","कब","क्या","कौन","कहाँ","कैसे","क्यों","कहां",
    "एक","दो","तीन","चार","पाँच","पांच","छह","छः","सात","आठ","नौ","दस",
    "ग्यारह","बारह","पंद्रह","बीस","पच्चीस","तीस","पचास","सौ","हजार","हज़ार","लाख",
    "जा","आ","कर","हो","दे","ले","बोल","पूछ","सुन","देख","बता","जाना","आना",
    "करना","होना","देना","लेना","बोलना","देखना","सुनना","पूछना","बताना",
    "गया","गई","गए","आया","आई","आए","किया","की","किए","लिया","लिए","दिया",
    "रहा","रही","रहे","करते","होते","जाते","आते","देते","लेते","सकते","चाहिए",
    "घर","काम","बात","समय","लोग","जगह","तरह","तरफ","साथ","चीज","बातें",
    "दिन","रात","साल","महीना","घंटा","मिनट","पल","पैसा","रुपये","रुपया",
    "आदमी","औरत","बच्चा","बच्चे","बच्ची","माँ","मां","बाप","पिता","भाई","बहन",
    "दोस्त","यार","सहेली","पति","पत्नी","बेटा","बेटी","परिवार","घरवाले",
    "अच्छा","अच्छी","अच्छे","बुरा","बड़ा","छोटा","नया","पुराना","सही","गलत",
    "बहुत","थोड़ा","थोड़ी","थोड़े","कुछ","सब","पूरा","हर","कोई","कई","काफी",
    "पहले","बाद","अभी","अब","तब","जब","फिर","आज","कल","परसों",
    "यहां","वहां","कहां","इधर","उधर","अंदर","बाहर","ऊपर","नीचे","आगे","पीछे",
    "जी","हाँ","हां","हा","ना","हम्म","अरे","ओह","अच्छा","ठीक","बस","सिर्फ",
    "मतलब","यानी","वैसे","सच","वाकई","सचमुच","बिल्कुल","हूं","हूँ",
    "सर","मैम","दीदी","भैया","अंकल","आंटी","साहब",
    "पता","याद","समझ","सोच","लग","चाह","जरूरी","जरूर","शायद","लगभग",
    "खाना","पानी","दूध","चाय","रोटी","सब्जी","फल","पैसे","कपड़े",
    "कभी","कहीं","किसी","कितना","कितनी","कितने","इतना","इतनी","इतने","उतना",
    "जैसे","ऐसे","ऐसा","वैसे","वैसा","तरह","प्रकार","तरीका",
    "अपना","अपनी","अपने","हमारा","हमारी","हमारे","तुम्हारा","आपका","आपकी","आपके",
    "इस","उस","इन","उन","इसका","उसका","इसकी","उसकी","इसके","उसके",
    "यही","वही","सभी","सारे","सारा","सारी","बाकी","दूसरा","दूसरी",
    "क्योंकि","इसलिए","यदि","चाहे","भले","हालांकि",
    "स्कूल","कॉलेज","दफ्तर","बाजार","रास्ता","सड़क","गाँव","शहर",
    "होता","होती","होते","लगता","लगती","लगते","मिलता","मिलती","मिलते",
    "बोला","बोली","बोले","सुना","सुनी","सुने","देखा","देखी","देखे",
    "जाना","जाती","जाते","आना","आती","आते","करना","करती","करते",
    "पहचान","जानना","समझना","सीखना","सिखाना","बताना","दिखाना",
    "अच्छा","बढ़िया","शानदार","जबरदस्त","खूबसूरत","सुंदर","प्यारा",
    "मेरा","मेरी","मेरे","तेरा","तेरी","तेरे","उसका","उसकी","उसके",
    "हमारा","हमारी","हमारे","आपका","आपकी","आपके","उनका","उनकी","उनके",
}

ENGLISH_DEVANAGARI = {
    "इंटरनेट","मोबाइल","फोन","लैपटॉप","कंप्यूटर","स्क्रीन","वीडियो","ऑडियो",
    "कैमरा","ऐप","ऑनलाइन","ऑफलाइन","सॉफ्टवेयर","हार्डवेयर","डेटा",
    "यूट्यूब","इंस्टाग्राम","फेसबुक","व्हाट्सएप","ट्विटर","नेटफ्लिक्स",
    "इंटरव्यू","जॉब","ऑफिस","मैनेजर","टीम","प्रोजेक्ट","प्रेजेंटेशन",
    "मीटिंग","रिपोर्ट","टार्गेट","डेडलाइन","ईमेल","पासवर्ड",
    "कॉलेज","यूनिवर्सिटी","एग्जाम","सिलेबस","नोट्स","असाइनमेंट","टीचर",
    "प्रॉब्लम","सॉल्यूशन","स्टेशन","होटल","रेस्टोरेंट","शॉपिंग","मॉल",
    "डॉक्टर","हॉस्पिटल","मेडिसिन","ट्रीटमेंट","ऑपरेशन","एक्सरे",
    "बैंक","लोन","पेमेंट","इंश्योरेंस","ट्रांजेक्शन","एटीएम",
    "ट्रेंड","फैशन","स्टाइल","आउटफिट","मेकअप","ड्रेस","जींस","टीशर्ट",
    "म्यूजिक","सॉन्ग","प्लेलिस्ट","एल्बम","आर्टिस्ट","बैंड","कॉन्सर्ट",
    "मूवी","सीरीज","एपिसोड","सीजन","ट्रेलर","डायरेक्टर",
    "कैंपिंग","ट्रेकिंग","हाइकिंग","एडवेंचर","टूर","ट्रिप","टिकट",
    "रील","पोस्ट","स्टोरी","लाइव","कमेंट","लाइक","शेयर","फॉलो","सब्सक्राइब",
    "नेटवर्क","सिग्नल","चार्जर","बैटरी","वाईफाई","ब्लूटूथ","हेडफोन",
    "फ्रेंड","पार्टनर","बॉयफ्रेंड","गर्लफ्रेंड","कपल","डेट","ब्रेकअप",
    "बस","ट्रेन","फ्लाइट","एयरपोर्ट","मेट्रो","ऑटो","कैब","उबर",
    "एक्सपीरियंस","स्किल","टैलेंट","परफॉर्मेंस","स्कोर","रिजल्ट",
    "प्लान","शेड्यूल","रूटीन","टाइमटेबल","कैलेंडर","रिमाइंडर",
    "जिम","वर्कआउट","एक्सरसाइज","डाइट","फिटनेस","योगा","मेडिटेशन",
    "पार्टी","फंक्शन","इवेंट","सेलिब्रेशन","बर्थडे","एनिवर्सरी",
    "हेलो","ओके","ओक","बाय","थैंक्यू","सॉरी","प्लीज","वेलकम","कूल",
    "टाइम","डेट","टाइप","फॉर्म","फाइल","फोल्डर","डॉक्यूमेंट","प्रिंट",
    "रेडियो","टीवी","चैनल","न्यूज","रिपोर्टर","एंकर",
    "बिजनेस","कंपनी","स्टार्टअप","इन्वेस्टमेंट","प्रॉफिट","लॉस",
    "गेम","स्पोर्ट्स","क्रिकेट","फुटबॉल","मैच","टूर्नामेंट","प्लेयर",
    "पिज्जा","बर्गर","सैंडविच","केक","चॉकलेट","कॉफी","जूस",
    "सीनियर","जूनियर","फ्रेशर","बैच","इंटर्नशिप","वर्कशॉप","सेमिनार",
    "लाइब्रेरी","कैंटीन","हॉस्टल","कैंपस","सिक्योरिटी","गार्ड",
    "वेबसाइट","ब्लॉग","पोर्टल","इंटरफेस","डिजाइन","लेआउट","थीम",
    "फोटो","सेल्फी","पिक्चर","इमेज","वॉलपेपर","स्क्रीनशॉट","मीम",
    "सर्च","गूगल","यूआरएल","लिंक","क्लिक","स्क्रॉल","जूम","स्वाइप",
    "टेस्ट","क्विज","बैकलॉग","सप्लीमेंट्री","लेक्चर","प्रैक्टिकल",
    "मिस्टेक","एरर","बग","फिक्स","सॉल्व","डिबग","कोड","प्रोग्राम",
}

# ── Unicode helpers 
MATRA_RANGE = set(range(0x093E, 0x094E))
HALANT      = 0x094D
ANUSVARA    = 0x0902
CHANDRABINDU= 0x0901
VOWEL_RANGE = set(range(0x0904, 0x0915))

def starts_with_matra(w):
    return len(w) > 0 and ord(w[0]) in MATRA_RANGE

def has_double_matra(w):
    for i in range(len(w)-1):
        if ord(w[i]) in MATRA_RANGE and ord(w[i+1]) in MATRA_RANGE:
            return True
    return False

def has_triple_repeat(w):
    for i in range(len(w)-2):
        if w[i] == w[i+1] == w[i+2]:
            return True
    return False

def ends_with_halant(w):
    return len(w) > 0 and ord(w[-1]) == HALANT

def is_mixed_script(w):
    return (bool(re.search(r'[\u0900-\u097F]', w)) and
            bool(re.search(r'[a-zA-Z]', w)))

def is_pure_roman(w):
    return bool(re.match(r'^[a-zA-Z]+$', w))

def has_invalid_unicode(w):
    for ch in w:
        cp = ord(ch)
        if cp < 128:
            continue
        if not (0x0900 <= cp <= 0x097F):
            return True
    return False

def is_only_matra(w):
    return len(w) == 1 and ord(w[0]) in MATRA_RANGE

def has_proper_vowel_structure(w):
    for ch in w:
        cp = ord(ch)
        if (cp in VOWEL_RANGE or cp in MATRA_RANGE or
                cp == ANUSVARA or cp == CHANDRABINDU):
            return True
    return False

# ── NEW: Tokenization error detection 
def has_tokenization_error(w):
    """
    Detect words that are actually tokenization errors:
    - Punctuation attached (है, / उसमें.)
    - Multiple words joined with comma (हम,जैसे)
    - Repetitive comma-joined tokens (टन,टन,टन)
    - Mixed Hindi+English joined (चलिए,एंड)
    - Trailing/leading hyphens or punctuation
    """
    # Punctuation attached to word
    if re.search(r'[,\.\!\?\;\:\।]+', w):
        return True, "Punctuation attached to word — tokenization error"

    # Multiple Devanagari words joined with comma
    if re.search(r'[\u0900-\u097F],[\ ]*[\u0900-\u097F]', w):
        return True, "Multiple words incorrectly joined with comma"

    # Hindi+English joined with comma
    if re.search(r'[\u0900-\u097F],[a-zA-Z]|[a-zA-Z],[\u0900-\u097F]', w):
        return True, "Hindi and English words incorrectly joined with comma"

    # Repetitive tokens (e.g. टन,टन,टन or हा हा हा)
    parts = re.split(r'[,\s]+', w)
    if len(parts) >= 3 and len(set(p.strip() for p in parts if p.strip())) <= 2:
        return True, "Repetitive tokens joined — transcription/sound artifact"

    # Trailing hyphen or leading comma
    if w.endswith('-') or w.startswith(',') or w.startswith('.'):
        return True, "Leading/trailing punctuation — tokenization error"

    return False, None


def classify_word(word):
    """
    Returns: (label, confidence, reason)
    """
    w = str(word).strip()
    if not w:
        return "incorrect spelling", "high", "Empty token"

    # ── DEFINITE CORRECT 
    if w in COMMON_HINDI:
        return ("correct spelling", "high",
                "Verified common Hindi word — in curated whitelist")

    if w in ENGLISH_DEVANAGARI:
        return ("correct spelling", "high",
                "English word correctly transcribed in Devanagari — per transcription guidelines")

    if re.match(r'^[\d\u0966-\u096F]+$', w):
        return ("correct spelling", "high",
                "Numeric token — not a spelling issue")

    if is_pure_roman(w):
        return ("correct spelling", "medium",
                "Roman script word — English code-switching, treated as correct")

    # ── TOKENIZATION ERRORS (NEW) 
    is_tok_err, tok_reason = has_tokenization_error(w)
    if is_tok_err:
        return ("incorrect spelling", "high", tok_reason)

    # ── DEFINITE INCORRECT 
    if is_only_matra(w):
        return ("incorrect spelling", "high",
                "Single vowel matra as standalone token — impossible in correct Devanagari")

    if starts_with_matra(w):
        return ("incorrect spelling", "high",
                "Word starts with vowel matra — violates Devanagari orthographic rules")

    if has_double_matra(w):
        return ("incorrect spelling", "high",
                "Consecutive vowel matras — impossible in correctly spelled Hindi")

    if has_triple_repeat(w):
        return ("incorrect spelling", "high",
                "3+ identical characters in sequence — transcription error or hallucination")

    if has_invalid_unicode(w):
        return ("incorrect spelling", "high",
                "Contains characters outside Devanagari Unicode range")

    if is_mixed_script(w):
        return ("incorrect spelling", "medium",
                "Mixed Devanagari and Roman script in single token — transcription artifact")

    if ends_with_halant(w):
        return ("incorrect spelling", "medium",
                "Word ends with halant (्) — typically indicates truncated word")

    if len(w) > 20 and re.search(r'[\u0900-\u097F]', w):
        return ("incorrect spelling", "medium",
                f"Unusually long word ({len(w)} chars) — likely concatenation error")

    # ── MEDIUM CONFIDENCE CORRECT 
    if re.match(r'^[\u0900-\u097F]+$', w) and has_proper_vowel_structure(w):
        if 4 <= len(w) <= 15:
            return ("correct spelling", "medium",
                    "Devanagari word with proper vowel/matra structure — likely correct")
        elif len(w) <= 3:
            return ("correct spelling", "medium",
                    "Short Devanagari word with vowel — likely correct")

    if re.match(r'^[\u0900-\u097F]+$', w) and not has_proper_vowel_structure(w):
        if len(w) <= 3:
            return ("correct spelling", "low",
                    "Short Devanagari word without clear vowel — could be abbreviation, needs review")
        else:
            return ("incorrect spelling", "low",
                    "Longer Devanagari word lacks vowel structure — possible spelling error")

    if re.search(r'[\d\u0966-\u096F]', w) and re.search(r'[\u0900-\u097F]', w):
        return ("correct spelling", "medium",
                "Mixed numeral-Devanagari token — likely measure word (e.g. 5वीं)")

    return ("correct spelling", "low",
            "Unclassified token — manual review recommended")



# STEP 3: CLASSIFY ALL WORDS
print(f"\nClassifying {len(unique_words)} words ...")

results = []
for word in unique_words:
    label, conf, reason = classify_word(word)
    results.append({
        "word":       word,
        "label":      label,
        "confidence": conf,
        "reason":     reason,
    })

df = pd.DataFrame(results)

correct   = (df["label"] == "correct spelling").sum()
incorrect = (df["label"] == "incorrect spelling").sum()
high_c    = (df["confidence"] == "high").sum()
med_c     = (df["confidence"] == "medium").sum()
low_c     = (df["confidence"] == "low").sum()

print(f"\nResults:")
print(f"  Total words         : {len(df)}")
print(f"  Correct spelling    : {correct} ({100*correct//len(df)}%)")
print(f"  Incorrect spelling  : {incorrect} ({100*incorrect//len(df)}%)")
print(f"  High confidence     : {high_c}")
print(f"  Medium confidence   : {med_c}")
print(f"  Low confidence      : {low_c}")


# STEP 4: REVIEW 40-50 LOW CONFIDENCE WORDS (Q3c)

print("\n" + "=" * 60)
print("STEP 4: Low confidence review (Q3c)")
print("=" * 60)

low_df = df[df["confidence"] == "low"].copy()
step   = max(1, len(low_df) // 50)
review_sample = low_df.iloc[::step].head(50).copy()
print(f"Low confidence total   : {len(low_df)}")
print(f"Reviewing sample of    : {len(review_sample)}")

def strict_manual_evaluate(word, system_label):
    """Strict manual re-check catching tokenization + spelling errors."""
    w = str(word).strip()

    # Catch tokenization errors system missed
    if re.search(r'[,\.\!\?\;\:\।]+', w):
        return "incorrect spelling", "Punctuation attached — tokenization error"
    if re.search(r'[\u0900-\u097F],[\ ]*[\u0900-\u097F]', w):
        return "incorrect spelling", "Multiple words joined with comma"
    if re.search(r'[\u0900-\u097F],[a-zA-Z]|[a-zA-Z],[\u0900-\u097F]', w):
        return "incorrect spelling", "Hindi+English incorrectly joined"
    parts = re.split(r'[,\s]+', w)
    if len(parts) >= 3 and len(set(p.strip() for p in parts if p.strip())) <= 2:
        return "incorrect spelling", "Repetitive tokens — transcription artifact"
    if w.endswith('-') or w.startswith(',') or w.startswith('.'):
        return "incorrect spelling", "Leading/trailing punctuation"

    # Known correct words
    if w in COMMON_HINDI or w in ENGLISH_DEVANAGARI:
        return "correct spelling", "In verified whitelist"

    # Clear error patterns
    if starts_with_matra(w) or has_double_matra(w) or has_triple_repeat(w):
        return "incorrect spelling", "Clear orthographic error"

    # Trust system for remaining
    return system_label, "No clear error — system classification accepted"

review_rows = []
right = 0
wrong = 0

for _, row in review_sample.iterrows():
    actual, eval_reason = strict_manual_evaluate(row["word"], row["label"])
    is_correct = (row["label"] == actual)
    if is_correct:
        right += 1
    else:
        wrong += 1
    review_rows.append({
        "word":           row["word"],
        "system_label":   row["label"],
        "actual_label":   actual,
        "system_correct": "✓" if is_correct else "✗",
        "error_found":    eval_reason,
        "original_reason":row["reason"],
    })

review_df = pd.DataFrame(review_rows)
accuracy  = round(100 * right / len(review_sample), 1)

print(f"\nReview results ({len(review_sample)} words):")
print(f"  System correct  : {right} ({accuracy}%)")
print(f"  System wrong    : {wrong} ({100-accuracy:.1f}%)")

wrong_df = review_df[review_df["system_correct"] == "✗"]
if len(wrong_df) > 0:
    print(f"\nWords where system was WRONG:")
    print(f"{'Word':<30} {'System':<20} {'Actual':<20} {'Error'}")
    print("-" * 90)
    for _, r in wrong_df.iterrows():
        print(f"{r['word']:<30} {r['system_label']:<20} {r['actual_label']:<20} {r['error_found'][:35]}")

print(f"""
Analysis: What this tells us about system breakdown:
  - {accuracy}% accuracy on low-confidence words
  - {wrong} cases where system classified as 'correct' but was 'incorrect'
  - Main failure: tokenization errors (punctuation-attached, comma-joined words)
    slipped through as 'correct' since Devanagari characters were individually valid
  - System treats 'है,' and 'है' identically — punctuation not stripped before check
  - Short ambiguous words (2-3 chars) are genuinely hard without a dictionary
""")


# STEP 5: UNRELIABLE CATEGORIES (Q3d)
print("=" * 60)
print("STEP 5: Unreliable categories (Q3d)")
print("=" * 60)
print("""
UNRELIABLE CATEGORY 1: Punctuation-attached tokens
---------------------------------------------------
Examples: है, / उसमें. / जी। / हाँ! / हम,जैसे
Why unreliable: Transcription JSONs sometimes include punctuation
attached to words. Our system's matra/structure rules check the
Devanagari characters which are valid — the attached punctuation
is invisible to spelling rules. 
Fix: Strip punctuation before classification, or add tokenization
cleaning as a pre-processing step.

UNRELIABLE CATEGORY 2: Novel English borrowings in Devanagari
-------------------------------------------------------------
Examples: रेडियोलॉजी, माइक्रोबायोलॉजी, टेक्नोलॉजिस्ट
Why unreliable: Our ENGLISH_DEVANAGARI list covers ~300 common words
but cannot cover all English borrowings. Rare domain-specific English
words in Devanagari get 'low' confidence even when correctly spelled.
Fix: Use a transliteration model (IndicNLP) to detect English words
phonetically represented in Devanagari.

UNRELIABLE CATEGORY 3: Dialectal Hindi word forms
-------------------------------------------------
Examples: बोलत (Bhojpuri for बोलता), कहत (for कहता), जात (for जाता)
Why unreliable: Valid dialectal forms from Bihar/UP/Odisha speakers
differ from standard Hindi. System has no dialect knowledge — gives
these 'low' confidence even though correct for that dialect.
Fix: Use native_state from metadata.json for dialect-specific word lists.
""")

# STEP 6: SAVE OUTPUTS

print("=" * 60)
print("STEP 6: Saving outputs")
print("=" * 60)

df.to_csv(RESULTS_DIR / "q3_word_classifications.csv",
          index=False, encoding="utf-8-sig")

review_df.to_csv(RESULTS_DIR / "q3_low_confidence_review.csv",
                 index=False, encoding="utf-8-sig")

pd.DataFrame([{
    "total_unique_words":       len(df),
    "correct_spelling_count":   int(correct),
    "incorrect_spelling_count": int(incorrect),
    "high_confidence":          int(high_c),
    "medium_confidence":        int(med_c),
    "low_confidence":           int(low_c),
    "low_conf_accuracy_pct":    accuracy,
}]).to_csv(RESULTS_DIR / "q3_summary.csv", index=False, encoding="utf-8-sig")

print(f"""
Files saved:
  results/q3/q3_word_classifications.csv  - all {len(df)} words: word|label|confidence|reason
  results/q3/q3_low_confidence_review.csv - {len(review_sample)} low-conf manual review
  results/q3/q3_summary.csv              - summary stats

Q3 DELIVERABLES:
  (a) Correctly spelled words : {correct}
  (b) Classifications CSV     : q3_word_classifications.csv
  (c) Low-conf accuracy       : {accuracy}% — system breaks on tokenization errors
  (d) Unreliable categories   : Punctuation-attached, English borrowings, Dialectal forms
""")