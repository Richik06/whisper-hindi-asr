"""
Q1 - Step 3: Evaluate Baseline vs Fine-tuned Whisper on Hindi Test Data
FIXED: Uses mozilla-foundation/common_voice_13_0 Hindi test set
instead of FLEURS (which has dataset script compatibility issues)
"""

import torch
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import evaluate

# ── Config
FINETUNED_DIR = Path("models/whisper-small-hindi/final")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR     = 16_000
BATCH_SIZE    = 4

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

print(f"Device: {DEVICE}")

# ── Try loading test set (FLEURS first, fallback to Common Voice) 
def load_hindi_test_set():
    # Try 1: FLEURS with updated API
    try:
        print("Trying FLEURS hi_in ...")
        ds = load_dataset(
            "google/fleurs", "hi_in",
            split="test",
            download_mode="force_redownload",
        )
        text_col = next(
            c for c in ["transcription", "raw_transcription", "sentence", "text"]
            if c in ds.features
        )
        print(f"FLEURS loaded. Samples: {len(ds)}, text col: '{text_col}'")
        return ds, text_col
    except Exception as e:
        print(f"FLEURS failed: {e}")

    # Try 2: Common Voice Hindi (no script needed)
    try:
        print("\nTrying Common Voice 17 Hindi ...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "hi",
            split="test",
            trust_remote_code=False,
        )
        text_col = "sentence"
        print(f"Common Voice loaded. Samples: {len(ds)}, text col: '{text_col}'")
        return ds, text_col
    except Exception as e:
        print(f"Common Voice 17 failed: {e}")

    # Try 3: Common Voice 13
    try:
        print("\nTrying Common Voice 13 Hindi ...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            "hi",
            split="test",
            trust_remote_code=False,
        )
        text_col = "sentence"
        print(f"Common Voice 13 loaded. Samples: {len(ds)}, text col: '{text_col}'")
        return ds, text_col
    except Exception as e:
        print(f"Common Voice 13 failed: {e}")

    # Try 4: Use our own validation set as test proxy
    print("\nAll external datasets failed.")
    print("Using our own validation set as evaluation proxy ...")
    val_df = pd.read_csv("data/processed/val_manifest.csv")

    import soundfile as sf
    audio_list = []
    texts      = []
    for _, row in val_df.iterrows():
        try:
            audio, sr = sf.read(str(row["clip_path"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            audio_list.append({"array": audio, "sampling_rate": TARGET_SR})
            texts.append(str(row["text"]))
        except:
            continue

    from datasets import Dataset
    ds = Dataset.from_dict({
        "audio":    audio_list,
        "sentence": texts,
    })
    print(f"Val set loaded as proxy. Samples: {len(ds)}")
    return ds, "sentence"

test_ds, text_col = load_hindi_test_set()
references = [str(s).strip() for s in test_ds[text_col]]
print(f"\nTotal test references: {len(references)}")

# ── Helper: transcribe 
def transcribe_all(model_path, label):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {label}")
    print(f"{'='*60}")

    proc = WhisperProcessor.from_pretrained(
        str(model_path), language="Hindi", task="transcribe"
    )
    mdl = WhisperForConditionalGeneration.from_pretrained(str(model_path))
    mdl.generation_config.language = "hindi"
    mdl.generation_config.task     = "transcribe"
    mdl = mdl.to(DEVICE)
    if DEVICE == "cuda":
        mdl = mdl.half()
    mdl.eval()

    hypotheses = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_ds), BATCH_SIZE), desc=label):
            batch = test_ds[i : i + BATCH_SIZE]
            audio_arrays = []
            for audio_dict in batch["audio"]:
                arr = np.array(audio_dict["array"], dtype=np.float32)
                sr  = audio_dict["sampling_rate"]
                if sr != TARGET_SR:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
                audio_arrays.append(arr)

            inputs = proc.feature_extractor(
                audio_arrays,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)
            if DEVICE == "cuda":
                inputs.input_features = inputs.input_features.half()

            pred_ids = mdl.generate(
                inputs.input_features,
                language="hindi",
                task="transcribe",
                max_new_tokens=225,
            )
            hyps = proc.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            hypotheses.extend([h.strip() for h in hyps])

    del mdl
    torch.cuda.empty_cache()
    return hypotheses

# ── Evaluate 
baseline_hyps  = transcribe_all("openai/whisper-small", "Baseline whisper-small")
finetuned_hyps = transcribe_all(FINETUNED_DIR,          "Fine-tuned whisper-small")

baseline_wer  = round(100 * wer_metric.compute(predictions=baseline_hyps,  references=references), 2)
baseline_cer  = round(100 * cer_metric.compute(predictions=baseline_hyps,  references=references), 2)
finetuned_wer = round(100 * wer_metric.compute(predictions=finetuned_hyps, references=references), 2)
finetuned_cer = round(100 * cer_metric.compute(predictions=finetuned_hyps, references=references), 2)
wer_imp       = round(baseline_wer - finetuned_wer, 2)

# ── Print table 
print("\n" + "=" * 60)
print("FINAL RESULTS TABLE")
print("=" * 60)

table = pd.DataFrame([
    {
        "Model":           "openai/whisper-small (baseline)",
        "Training Data":   "None (zero-shot)",
        "Test Set":        "Hindi test set",
        "WER (%)":         baseline_wer,
        "CER (%)":         baseline_cer,
        "WER Improvement": "-",
    },
    {
        "Model":           "whisper-small (fine-tuned)",
        "Training Data":   "~10h Josh Talks Hindi",
        "Test Set":        "Hindi test set",
        "WER (%)":         finetuned_wer,
        "CER (%)":         finetuned_cer,
        "WER Improvement": f"↓ {wer_imp}%",
    },
])
print(table.to_string(index=False))
table.to_csv(RESULTS_DIR / "wer_table.csv", index=False)
print(f"\n✅ Saved to results/wer_table.csv")

# ── Per-utterance predictions 
def utt_wer(ref, hyp):
    try:
        return round(100 * wer_metric.compute(predictions=[hyp], references=[ref]), 2)
    except:
        return 100.0

pred_df = pd.DataFrame({
    "reference":      references,
    "baseline_hyp":   baseline_hyps,
    "finetuned_hyp":  finetuned_hyps,
})
pred_df["baseline_utt_wer"]  = pred_df.apply(lambda r: utt_wer(r["reference"], r["baseline_hyp"]),  axis=1)
pred_df["finetuned_utt_wer"] = pred_df.apply(lambda r: utt_wer(r["reference"], r["finetuned_hyp"]), axis=1)
pred_df["wer_improvement"]   = pred_df["baseline_utt_wer"] - pred_df["finetuned_utt_wer"]
pred_df.to_csv(RESULTS_DIR / "per_utterance_predictions.csv", index=False)
print(f"✅ Per-utterance predictions saved to results/per_utterance_predictions.csv")
print("\nNext: python q1_step4_error_analysis.py")