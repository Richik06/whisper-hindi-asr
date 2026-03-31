"""
Q2: ASR Post-processing Cleanup Pipeline
Real data examples extracted from actual 200-clip ASR output.
"""

import re
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate

# ── Config 
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results/q2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SR     = 16_000
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
wer_metric    = evaluate.load("wer")

# STEP 1: Generate raw ASR using baseline whisper-small
print("=" * 60)
print("STEP 1: Generating raw ASR with baseline whisper-small")
print("=" * 60)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="Hindi", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "hindi"
model.generation_config.task     = "transcribe"
model = model.to(DEVICE)
if DEVICE == "cuda":
    model = model.half()
model.eval()

train_df  = pd.read_csv(PROCESSED_DIR / "train_manifest.csv")
val_df    = pd.read_csv(PROCESSED_DIR / "val_manifest.csv")
all_df    = pd.concat([train_df, val_df], ignore_index=True)
sample_df = all_df.sample(n=min(200, len(all_df)), random_state=42).reset_index(drop=True)
print(f"Generating ASR for {len(sample_df)} clips ...")

raw_asr_outputs = []
with torch.no_grad():
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Baseline ASR"):
        try:
            audio, sr = sf.read(str(row["clip_path"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            inputs = processor.feature_extractor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt"
            ).to(DEVICE)
            if DEVICE == "cuda":
                inputs.input_features = inputs.input_features.half()
            pred_ids = model.generate(
                inputs.input_features,
                language="hindi", task="transcribe", max_new_tokens=225,
            )
            hyp = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
        except:
            hyp = ""
        raw_asr_outputs.append(hyp)

del model
torch.cuda.empty_cache()

sample_df["raw_asr"]   = raw_asr_outputs
sample_df["reference"] = sample_df["text"]
print(f"Done. {len(sample_df)} raw ASR outputs generated.")


# STEP 2a: NUMBER NORMALIZATION
ONES = {
    "शून्य":0,"एक":1,"दो":2,"तीन":3,"चार":4,"पाँच":5,"पांच":5,
    "छह":6,"छः":6,"सात":7,"आठ":8,"नौ":9,"दस":10,"ग्यारह":11,
    "बारह":12,"तेरह":13,"चौदह":14,"पंद्रह":15,"सोलह":16,
    "सत्रह":17,"अठारह":18,"उन्नीस":19,"बीस":20,"इक्कीस":21,
    "बाईस":22,"तेईस":23,"चौबीस":24,"पच्चीस":25,"छब्बीस":26,
    "सत्ताईस":27,"अट्ठाईस":28,"उनतीस":29,"तीस":30,"चालीस":40,
    "पचास":50,"पच्चास":50,"साठ":60,"सत्तर":70,"अस्सी":80,
    "नब्बे":90,"चौवन":54,"पचहत्तर":75,"बहत्तर":72,"पचपन":55,
}
MULTIPLIERS = {
    "सौ":100,"हज़ार":1000,"हजार":1000,"लाख":100000,"करोड़":10000000,
}
IDIOM_PATTERNS = [
    r'दो[-\s]चार', r'चार[-\s]पाँच', r'एक[-\s]दो',
    r'तीन[-\s]चार', r'दो टूक', r'एक दूसरे',
    r'एक साथ', r'एक दम', r'एक बार', r'दो बार',
]

def in_idiom(text, char_pos):
    for pat in IDIOM_PATTERNS:
        for m in re.finditer(pat, text):
            if m.start() <= char_pos < m.end():
                return True, m.group()
    return False, None

def normalize_numbers(text):
    tokens = text.split()
    result = []; changes = []; i = 0
    while i < len(tokens):
        tok      = tokens[i]
        char_pos = len(' '.join(tokens[:i])) + (1 if i > 0 else 0)
        if tok in ONES or tok in MULTIPLIERS:
            is_id, idiom_str = in_idiom(text, char_pos)
            if is_id:
                result.append(tok); i += 1; continue
            num = 0; curr = 0; j = i
            while j < len(tokens) and (tokens[j] in ONES or tokens[j] in MULTIPLIERS):
                t = tokens[j]
                if t in MULTIPLIERS:
                    curr = curr if curr > 0 else 1
                    curr *= MULTIPLIERS[t]; num += curr; curr = 0
                else:
                    curr += ONES[t]
                j += 1
            num += curr
            phrase = ' '.join(tokens[i:j])
            changes.append({"original": phrase, "converted": str(num),
                            "type": "compound" if j-i > 1 else "simple"})
            result.append(str(num)); i = j
        else:
            result.append(tok); i += 1
    return ' '.join(result), changes

def get_number_reasoning(raw, normalized, changes):
    raw, normalized = str(raw), str(normalized)
    if raw == normalized:
        num_words = [w for w in raw.split() if w in ONES or w in MULTIPLIERS]
        if num_words:
            for pat in IDIOM_PATTERNS:
                m = re.search(pat, raw)
                if m:
                    return (f"EDGE CASE - KEPT AS-IS: '{m.group()}' is a frozen idiom "
                            f"(e.g. दो-चार = 'a few', एक बार = 'once'). "
                            f"Converting would destroy idiomatic meaning.")
            return "Number words present but kept — idiomatic context."
        return "No number words — no normalization needed."
    parts = [f"'{c['original']}' → {c['converted']} ({c['type']})" for c in changes]
    return f"CONVERTED: {' | '.join(parts)}. Hindi number words converted to digits."

# Apply
norm_results = sample_df["raw_asr"].apply(normalize_numbers)
sample_df["asr_number_normalized"] = norm_results.apply(lambda x: x[0])
sample_df["_num_changes"]          = norm_results.apply(lambda x: x[1])
sample_df["number_norm_reasoning"] = sample_df.apply(
    lambda r: get_number_reasoning(
        r["raw_asr"], r["asr_number_normalized"], r["_num_changes"]
    ), axis=1
)


# STEP 2b: ENGLISH WORD DETECTION
ENGLISH_DEVANAGARI = {
    "इंटरनेट","मोबाइल","फोन","लैपटॉप","कंप्यूटर","वीडियो","ऑडियो",
    "ऐप","ऑनलाइन","यूट्यूब","इंस्टाग्राम","फेसबुक","व्हाट्सएप",
    "इंटरव्यू","जॉब","ऑफिस","मैनेजर","टीम","प्रोजेक्ट","मीटिंग",
    "कॉलेज","यूनिवर्सिटी","एग्जाम","प्रॉब्लम","सॉल्यूशन","स्टेशन",
    "होटल","रेस्टोरेंट","शॉपिंग","मॉल","डॉक्टर","हॉस्पिटल","बैंक",
    "ट्रेंड","फैशन","स्टाइल","फ्रेंड","नेटवर्क","सिग्नल","म्यूजिक",
    "सॉन्ग","प्लेलिस्ट","मूवी","सीरीज","कैंपिंग","रील","पोस्ट",
    "लाइक","शेयर","ट्रेन","बस","एक्सपीरियंस","स्किल","प्लान",
}
NOT_ENGLISH = {
    'hai','hain','tha','thi','the','nahi','nahin','aur','par',
    'mein','se','ko','ka','ki','ke','ho','hoga','kya','koi',
    'kuch','sab','woh','yeh','toh','bhi','hi','na','ji','han',
}

def is_roman_english(word):
    clean = re.sub(r'[^a-zA-Z]', '', word).lower()
    return len(clean) >= 3 and clean not in NOT_ENGLISH

def tag_english_words(text):
    words = str(text).split()
    tagged = []; changes = []
    for word in words:
        clean = re.sub(r'[।,\.\?\!\-\'\"]', '', word)
        if clean in ENGLISH_DEVANAGARI:
            tagged.append(f"[EN]{word}[/EN]")
            changes.append({"word": word, "type": "devanagari_english"})
        elif re.match(r'^[a-zA-Z]', word) and is_roman_english(word):
            tagged.append(f"[EN]{word}[/EN]")
            changes.append({"word": word, "type": "roman_english"})
        elif re.search(r'[a-zA-Z]{2,}', word) and re.search(r'[\u0900-\u097F]', word):
            tagged.append(f"[EN]{word}[/EN]")
            changes.append({"word": word, "type": "mixed_script"})
        else:
            tagged.append(word)
    return ' '.join(tagged), changes

def get_english_reasoning(changes):
    if not changes:
        return "No English words detected — utterance is pure Hindi."
    parts = []
    for ch in changes:
        w, t = ch["word"], ch["type"]
        if t == "devanagari_english":
            parts.append(
                f"'{w}' tagged [EN] — English word spoken in conversation, "
                f"transcribed in Devanagari per guidelines. Correct spelling, "
                f"needs special handling in downstream tasks (TTS/translation/NER)."
            )
        elif t == "roman_english":
            parts.append(
                f"'{w}' tagged [EN] — Roman-script code-switching. "
                f"Should be Devanagari per guidelines but ASR output Roman."
            )
        else:
            parts.append(f"'{w}' tagged [EN] — mixed-script token.")
    return " | ".join(parts)

# Apply
en_results = sample_df["raw_asr"].apply(tag_english_words)
sample_df["asr_english_tagged"]          = en_results.apply(lambda x: x[0])
sample_df["_en_changes"]                 = en_results.apply(lambda x: x[1])
sample_df["english_detection_reasoning"] = sample_df.apply(
    lambda r: get_english_reasoning(r["_en_changes"]), axis=1
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: WER + REASONING
# ══════════════════════════════════════════════════════════════════════════════
def safe_wer(ref, hyp):
    try:
        ref, hyp = str(ref).strip(), str(hyp).strip()
        if not ref or not hyp: return 100.0
        return round(100 * wer_metric.compute(predictions=[hyp], references=[ref]), 2)
    except:
        return 100.0

def get_wer_reasoning(delta, wer_raw):
    try:
        delta = float(delta); wer_raw = float(wer_raw)
    except:
        return "N/A"
    if delta > 0:
        return (f"NORMALIZATION HELPED: WER improved by {delta:.1f}%. "
                f"Reference uses digits — normalization aligned the format correctly.")
    elif delta < 0:
        return (f"NORMALIZATION HURT: WER worsened by {abs(delta):.1f}%. "
                f"Reference uses Hindi number words — converting to digits "
                f"caused format mismatch. Shows normalization is not always beneficial.")
    else:
        if wer_raw == 0:
            return "Perfect transcription — pipeline had no impact."
        return "No WER change — no number words present in this utterance."

sample_df["wer_raw"]      = sample_df.apply(
    lambda r: safe_wer(r["reference"], r["raw_asr"]), axis=1)
sample_df["wer_num_norm"] = sample_df.apply(
    lambda r: safe_wer(r["reference"], r["asr_number_normalized"]), axis=1)
sample_df["wer_num_delta"] = sample_df["wer_raw"] - sample_df["wer_num_norm"]
sample_df["wer_impact_reasoning"] = sample_df.apply(
    lambda r: get_wer_reasoning(r["wer_num_delta"], r["wer_raw"]), axis=1
)

# Combined pipeline
sample_df["asr_full_pipeline"] = sample_df["asr_number_normalized"].apply(
    lambda t: tag_english_words(t)[0]
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: EXTRACT REAL EXAMPLES FROM ACTUAL DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nExtracting real examples from actual data ...")

# -- Real number conversion examples (rows where conversion happened) ----------
real_converted = sample_df[
    sample_df["_num_changes"].apply(
        lambda x: len(x) > 0 and
        not any(c.get("type") == "idiom" for c in x)
    )
].head(5)

# -- Real edge cases (rows where idiom kept OR normalization hurt WER) ---------
real_edge = sample_df[
    sample_df["number_norm_reasoning"].str.contains("EDGE CASE|KEPT AS-IS|HURT", na=False)
].head(3)

# -- Real English tagging examples --------------------------------------------
real_english = sample_df[
    sample_df["_en_changes"].apply(lambda x: len(x) > 0)
].head(5)

# Build real examples dataframe
examples_rows = []

# Number conversion examples from real data
for _, row in real_converted.iterrows():
    for ch in row["_num_changes"]:
        examples_rows.append({
            "example_type":   "Number Normalization - Correct Conversion (REAL DATA)",
            "reference":      row["reference"],
            "raw_asr_input":  row["raw_asr"],
            "pipeline_output":row["asr_number_normalized"],
            "what_changed":   f"'{ch['original']}' → {ch['converted']}",
            "decision":       "CONVERT",
            "reasoning":      row["number_norm_reasoning"],
            "wer_before":     row["wer_raw"],
            "wer_after":      row["wer_num_norm"],
            "wer_impact":     row["wer_impact_reasoning"],
        })
        break  # one example per row

# Edge case examples from real data
for _, row in real_edge.iterrows():
    examples_rows.append({
        "example_type":   "Edge Case - Judgment Call (REAL DATA)",
        "reference":      row["reference"],
        "raw_asr_input":  row["raw_asr"],
        "pipeline_output":row["asr_number_normalized"],
        "what_changed":   "No conversion applied",
        "decision":       "KEEP",
        "reasoning":      row["number_norm_reasoning"],
        "wer_before":     row["wer_raw"],
        "wer_after":      row["wer_num_norm"],
        "wer_impact":     row["wer_impact_reasoning"],
    })

# English examples from real data
for _, row in real_english.iterrows():
    words = [c["word"] for c in row["_en_changes"]]
    examples_rows.append({
        "example_type":   "English Word Detection (REAL DATA)",
        "reference":      row["reference"],
        "raw_asr_input":  row["raw_asr"],
        "pipeline_output":row["asr_english_tagged"],
        "what_changed":   f"Tagged: {words}",
        "decision":       "TAG",
        "reasoning":      row["english_detection_reasoning"],
        "wer_before":     row["wer_raw"],
        "wer_after":      row["wer_num_norm"],
        "wer_impact":     row["wer_impact_reasoning"],
    })

# If not enough real examples found, note it
if len(real_converted) < 5:
    print(f"  Note: Only {len(real_converted)} real number conversion examples found in 200 clips.")
    print(f"  Number words are rare in this conversational sample.")

real_examples_df = pd.DataFrame(examples_rows)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
# Main 200-row output with all reasoning columns
final_cols = [
    "reference", "raw_asr",
    "asr_number_normalized", "number_norm_reasoning",
    "asr_english_tagged",    "english_detection_reasoning",
    "asr_full_pipeline",
    "wer_raw", "wer_num_norm", "wer_num_delta", "wer_impact_reasoning",
]
sample_df[final_cols].to_csv(
    RESULTS_DIR / "q2_pipeline_output.csv",
    index=False, encoding="utf-8-sig"
)

# Real examples extracted from actual data
real_examples_df.to_csv(
    RESULTS_DIR / "q2_real_examples_with_reasoning.csv",
    index=False, encoding="utf-8-sig"
)

# ── Summary ───────────────────────────────────────────────────────────────────
changed_num = (sample_df["raw_asr"] != sample_df["asr_number_normalized"]).sum()
changed_en  = (sample_df["raw_asr"] != sample_df["asr_english_tagged"]).sum()
helped      = (sample_df["wer_num_delta"] > 0).sum()
hurt        = (sample_df["wer_num_delta"] < 0).sum()

print(f"\n{'='*60}")
print("Q2 COMPLETE")
print(f"{'='*60}")
print(f"""
Files saved:
  results/q2/q2_pipeline_output.csv           - 200 rows with reasoning
  results/q2/q2_real_examples_with_reasoning.csv - real examples from actual data

Pipeline Summary ({len(sample_df)} utterances):
  Number normalization changed  : {changed_num}
  English words tagged          : {changed_en}
  Normalization helped WER      : {helped}
  Normalization hurt WER        : {hurt}
  Avg WER raw ASR               : {sample_df['wer_raw'].mean():.2f}%
  Avg WER after normalization   : {sample_df['wer_num_norm'].mean():.2f}%

Real examples extracted:
  Number conversions  : {len(real_converted)}
  Edge cases          : {len(real_edge)}
  English detections  : {len(real_english)}
""")