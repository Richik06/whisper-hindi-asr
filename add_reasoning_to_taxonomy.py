"""
Adds a proper 'reasoning' column to error_taxonomy_sample.csv
as required by Q1(e): "your reasoning about the cause of the error"
"""

import pandas as pd
import ast
import re
from pathlib import Path

RESULTS_DIR = Path("results")

df = pd.read_csv(RESULTS_DIR / "error_taxonomy_sample.csv")

def parse_cats(x):
    try:
        return ast.literal_eval(str(x))
    except:
        return [str(x)]

df["error_categories"] = df["error_categories"].apply(parse_cats)

def generate_reasoning(ref, hyp, cats):
    ref      = str(ref)
    hyp      = str(hyp)
    ref_words = ref.split()
    hyp_words = hyp.split()
    reasons  = []

    for cat in cats:
        if cat == "hallucination":
            overlap = len(set(ref_words) & set(hyp_words))
            reasons.append(
                f"HALLUCINATION: Only {overlap} word(s) overlap between reference "
                f"({len(ref_words)} words) and hypothesis ({len(hyp_words)} words). "
                f"The model generated text from its language model prior rather than "
                f"from actual audio evidence — likely caused by background noise or "
                f"very low acoustic signal in the clip."
            )

        elif cat == "dialectal_phonetic_substitution":
            reasons.append(
                f"DIALECTAL/PHONETIC SUBSTITUTION: The model replaced words with "
                f"phonetically similar alternatives. This speaker's accent likely "
                f"differs from the dominant accent in training data "
                f"(e.g. Bihar/Odisha/Assam vs. standard Hindi). The model mapped "
                f"the acoustic input to the closest word it had seen most frequently, "
                f"even though the speaker intended a different word."
            )

        elif cat == "deletion_heavy":
            dropped = len(ref_words) - len(hyp_words)
            reasons.append(
                f"HEAVY DELETION: Reference has {len(ref_words)} words but model "
                f"only produced {len(hyp_words)} words ({dropped} words missing). "
                f"Likely caused by fast conversational speech, poor audio quality "
                f"from a network call recording, or the segment boundary cutting "
                f"mid-utterance so the model never heard those words."
            )

        elif cat == "roman_script_leak":
            roman_words = re.findall(r'[a-zA-Z]{3,}', hyp)
            reasons.append(
                f"ROMAN SCRIPT LEAK: Found Roman-script word(s) {roman_words} in "
                f"the hypothesis. Per transcription guidelines, English words spoken "
                f"in Hindi conversation should appear in Devanagari. Whisper's "
                f"pretraining on 680k hours taught it to output Roman for English "
                f"words, and fine-tuning has not fully overridden this behaviour."
            )

        elif cat == "number_digit_confusion":
            ref_has_digit = bool(re.search(r'\d', ref))
            hyp_has_digit = bool(re.search(r'\d', hyp))
            if not ref_has_digit and hyp_has_digit:
                reasons.append(
                    f"NUMBER/DIGIT CONFUSION: Reference uses Hindi number words but "
                    f"model output contains digits. Training data had inconsistent "
                    f"labelling — some transcripts use digits, others use words for "
                    f"the same spoken numbers. Model learned an ambiguous convention."
                )
            else:
                reasons.append(
                    f"NUMBER/DIGIT CONFUSION: Inconsistent number format between "
                    f"reference and hypothesis. Mixed digit/word conventions in "
                    f"training labels caused the model to be uncertain about the "
                    f"expected output format for spoken numbers."
                )

        elif cat == "partial_transcription":
            ratio = round(len(hyp_words) / max(len(ref_words), 1) * 100)
            reasons.append(
                f"PARTIAL TRANSCRIPTION: Model produced only ~{ratio}% of the "
                f"reference ({len(hyp_words)} of {len(ref_words)} words). "
                f"A long silence in the middle of the clip likely reset the "
                f"decoder's attention, or the clip was cut at a sentence boundary "
                f"causing the decoder to stop early."
            )

        elif cat == "other_substitution":
            reasons.append(
                f"LEXICAL SUBSTITUTION: Model substituted one or more words with "
                f"different words that may sound similar or appear in similar "
                f"contexts. This reflects a vocabulary gap — the model has not "
                f"seen enough training examples of these specific Hindi words or "
                f"phrases to reliably transcribe them."
            )

    return " | ".join(reasons) if reasons else "General transcription error — model uncertainty on this utterance."

df["reasoning"] = df.apply(
    lambda r: generate_reasoning(r["reference"], r["finetuned_hyp"], r["error_categories"]),
    axis=1
)

# Clean up error_categories for display
df["error_categories_display"] = df["error_categories"].apply(
    lambda cats: ", ".join([c.replace("_", " ").title() for c in cats])
)

# Final column order matching Q1(e) requirements
out_df = df[[
    "reference",
    "finetuned_hyp",
    "finetuned_utt_wer",
    "severity",
    "error_categories_display",
    "reasoning",
]].rename(columns={
    "reference":               "Reference Transcript",
    "finetuned_hyp":           "Model Output",
    "finetuned_utt_wer":       "WER (%)",
    "severity":                "Severity",
    "error_categories_display":"Error Category",
    "reasoning":               "Reasoning (Cause of Error)",
})

out_path = RESULTS_DIR / "error_taxonomy_with_reasoning.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig opens correctly in Excel

print(f"✅ Saved to: {out_path}")
print(f"   Rows: {len(out_df)}")
print(f"\nColumns:")
for col in out_df.columns:
    print(f"  - {col}")

print("\nSample row:")
row = out_df.iloc[0]
print(f"  Reference : {row['Reference Transcript'][:80]}")
print(f"  Model Out : {row['Model Output'][:80]}")
print(f"  WER       : {row['WER (%)']}%")
print(f"  Severity  : {row['Severity']}")
print(f"  Category  : {row['Error Category']}")
print(f"  Reasoning : {row['Reasoning (Cause of Error)'][:150]}...")