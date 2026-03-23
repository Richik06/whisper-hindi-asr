"""
Generates a clean readable error taxonomy report from the results CSV
Save the terminal output of this script as your Q1(e) answer
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import ast

RESULTS_DIR = Path("results")

# Load the taxonomy sample
df = pd.read_csv(RESULTS_DIR / "error_taxonomy_sample.csv")
print(f"Loaded {len(df)} sampled error utterances\n")

# Parse error_categories column (stored as string list)
def parse_cats(x):
    try:
        return ast.literal_eval(str(x))
    except:
        return [str(x)]

df["error_categories"] = df["error_categories"].apply(parse_cats)

# Count categories
all_cats = [c for cats in df["error_categories"] for c in cats]
cat_counts = Counter(all_cats)

TAXONOMY = {
    "other_substitution": {
        "desc": "Word-level substitution where the model replaces a word with "
                "a different but sometimes phonetically similar word. Does not "
                "fit into more specific categories.",
        "cause": "General vocabulary gap — the model has not seen enough examples "
                 "of certain domain-specific or conversational Hindi words during "
                 "fine-tuning. The acoustic signal is close but the language model "
                 "component picks a more frequent word from training.",
    },
    "dialectal_phonetic_substitution": {
        "desc": "Model substitutes a word with a phonetically similar alternative "
                "due to dialectal variation. Speakers from Bihar, Odisha, Assam, UP "
                "pronounce the same Hindi word differently from standard Hindi.",
        "cause": "Training data is weighted toward certain accent regions. The model "
                 "has not heard enough samples from under-represented dialect zones "
                 "and defaults to the most frequent phonetic form it has seen.",
    },
    "deletion_heavy": {
        "desc": "Model drops entire words or phrases, producing a hypothesis that is "
                "significantly shorter than the reference transcript.",
        "cause": "Fast speech, overlapping speech in conversational data, or poor audio "
                 "quality from network call recordings. Segment boundaries may also cut "
                 "mid-word causing the decoder to miss the beginning of utterances.",
    },
    "hallucination": {
        "desc": "Model generates plausible-sounding but completely wrong Hindi text "
                "with very little word overlap with the reference.",
        "cause": "Background noise, very short clips (1-2 words), or low acoustic "
                 "signal causing the decoder to generate from language model prior "
                 "rather than from audio evidence.",
    },
    "roman_script_leak": {
        "desc": "English words spoken in Hindi conversation appear in Roman script "
                "in the hypothesis instead of Devanagari script as required by the "
                "transcription guidelines.",
        "cause": "Whisper's pretraining on 680k hours rewards Roman output for English "
                 "words. The fine-tuning data guideline requires Devanagari but the "
                 "model has not fully overridden this pretraining behaviour.",
    },
    "number_digit_confusion": {
        "desc": "Model outputs digits (2, 10, 100) where the reference has spelled-out "
                "Hindi number words, or vice versa.",
        "cause": "Inconsistency in training data labelling — some transcripts use digits "
                 "while others use Hindi number words for the same spoken content. "
                 "The model is confused about the expected output convention.",
    },
    "partial_transcription": {
        "desc": "Model transcribes approximately half of the utterance and produces "
                "a significantly truncated output.",
        "cause": "Long silence mid-clip resets the decoder attention, or clip starts "
                 "mid-sentence due to imprecise timestamp-based segmentation.",
    },
}

print("=" * 70)
print("Q1(e) ERROR TAXONOMY — Categories from observed data")
print("=" * 70)
print(f"\nTotal sampled utterances : {len(df)}")
print(f"Total error categories   : {len(cat_counts)}\n")

print("Category frequency:")
for cat, count in cat_counts.most_common():
    pct = round(100 * count / len(df), 1)
    print(f"  {cat:<40} n={count}  ({pct}%)")

print("\n" + "=" * 70)
print("DETAILED TAXONOMY WITH EXAMPLES")
print("=" * 70)

for rank, (cat, count) in enumerate(cat_counts.most_common(), 1):
    info = TAXONOMY.get(cat, {
        "desc": "General transcription error.",
        "cause": "Model uncertainty or out-of-domain content."
    })
    pct = round(100 * count / len(df), 1)

    print(f"\n{'─'*70}")
    print(f"CATEGORY {rank}: {cat.upper().replace('_',' ')}")
    print(f"Frequency: {count} utterances ({pct}% of error sample)")
    print(f"{'─'*70}")
    print(f"\nDescription:\n  {info['desc']}")
    print(f"\nCause:\n  {info['cause']}")

    # Get examples for this category
    examples = df[df["error_categories"].apply(lambda c: cat in c)].head(5)

    print(f"\nExamples ({min(len(examples), 5)} shown):")
    for i, (_, row) in enumerate(examples.iterrows(), 1):
        print(f"\n  Example {i}:")
        print(f"  Reference : {row['reference']}")
        print(f"  Model out : {row['finetuned_hyp']}")
        print(f"  WER       : {row['finetuned_utt_wer']}%  [{row['severity']}]")

        # Auto-reasoning based on category
        ref_words = str(row['reference']).split()
        hyp_words = str(row['finetuned_hyp']).split()
        if cat == "deletion_heavy":
            print(f"  Reasoning : Reference has {len(ref_words)} words, model only produced "
                  f"{len(hyp_words)} words ({len(ref_words)-len(hyp_words)} words dropped). "
                  f"Likely fast speech or poor audio quality.")
        elif cat == "hallucination":
            ref_set = set(ref_words)
            hyp_set = set(hyp_words)
            overlap = len(ref_set & hyp_set)
            print(f"  Reasoning : Only {overlap} word(s) overlap between reference and "
                  f"hypothesis. Model generated from language prior, not audio signal.")
        elif cat == "dialectal_phonetic_substitution":
            print(f"  Reasoning : Words are phonetically similar but lexically different, "
                  f"indicating accent/dialect mismatch between speaker and training data.")
        elif cat == "roman_script_leak":
            print(f"  Reasoning : Model output contains Roman script English words "
                  f"that should appear in Devanagari per transcription guidelines.")
        elif cat == "number_digit_confusion":
            print(f"  Reasoning : Inconsistent number format between reference and "
                  f"hypothesis due to mixed conventions in training labels.")
        elif cat == "partial_transcription":
            print(f"  Reasoning : Hypothesis ({len(hyp_words)} words) is significantly "
                  f"shorter than reference ({len(ref_words)} words) but not empty — "
                  f"suggests decoder stopped mid-utterance.")
        else:
            print(f"  Reasoning : Lexical substitution where model chose a different "
                  f"word than reference, likely due to vocabulary gap or similar "
                  f"acoustic pattern in training data.")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Category':<42} {'Count':>6} {'%':>6} {'Severity breakdown'}")
print("-" * 70)
for cat, count in cat_counts.most_common():
    pct = round(100 * count / len(df), 1)
    cat_rows = df[df["error_categories"].apply(lambda c: cat in c)]
    sev = cat_rows["severity"].value_counts().to_dict()
    sev_str = " | ".join([f"{k}:{v}" for k,v in sev.items()])
    print(f"{cat:<42} {count:>6} {pct:>5.1f}%  {sev_str}")

print("\n✅ Copy this entire output as your Q1(e) answer")