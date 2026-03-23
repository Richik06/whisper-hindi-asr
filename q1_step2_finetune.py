"""
Q1 - Step 2: Fine-tune Whisper-small
FIXED: Removed suppress_tokens and forced_decoder_ids conflicts
Works on Windows with RTX 4050 8GB
"""

import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

from torch.utils.data import Dataset as TorchDataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODEL_ID      = "openai/whisper-small"
OUTPUT_DIR    = Path("models/whisper-small-hindi")
TARGET_SR     = 16_000
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Custom PyTorch Dataset ─────────────────────────────────────────────────────
class HindiASRDataset(TorchDataset):
    def __init__(self, manifest_csv, processor):
        self.df        = pd.read_csv(manifest_csv)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio, sr = sf.read(str(row["clip_path"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        ).input_features[0]

        labels = self.processor.tokenizer(
            str(row["text"]),
            return_tensors="pt",
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels}

# ── Data collator ──────────────────────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

# ── Encoder freeze/unfreeze ───────────────────────────────────────────────────
def set_encoder_grad(m, requires_grad):
    for p in m.model.encoder.parameters():
        p.requires_grad = requires_grad

class UnfreezeEncoderCallback(TrainerCallback):
    def __init__(self, step=1000):
        self.step = step
        self.done = False
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self.done and state.global_step >= self.step:
            set_encoder_grad(model, True)
            self.done = True
            print(f"\nEncoder unfrozen at step {state.global_step}")

# ── Load processor and model ──────────────────────────────────────────────────
print("Loading processor ...")
processor = WhisperProcessor.from_pretrained(
    MODEL_ID, language="Hindi", task="transcribe"
)

print("Loading model ...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# FIXED: only set language/task on generation_config, nothing else
# Removed suppress_tokens and forced_decoder_ids which conflict with new API
model.generation_config.language = "hindi"
model.generation_config.task     = "transcribe"

model.config.use_cache = False
model.gradient_checkpointing_enable()
set_encoder_grad(model, False)

# ── Build datasets ─────────────────────────────────────────────────────────────
print("Building train dataset ...")
train_dataset = HindiASRDataset(PROCESSED_DIR / "train_manifest.csv", processor)
print(f"  Train samples: {len(train_dataset)}")

print("Building val dataset ...")
val_dataset = HindiASRDataset(PROCESSED_DIR / "val_manifest.csv", processor)
print(f"  Val samples:   {len(val_dataset)}")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ── Metrics ────────────────────────────────────────────────────────────────────
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [p.strip() for p in processor.tokenizer.batch_decode(
        pred_ids,  skip_special_tokens=True)]
    label_str = [l.strip() for l in processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True)]
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 2), "cer": round(cer, 2)}

# ── Training arguments ─────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 4,
    learning_rate               = 1e-5,
    warmup_steps                = 500,
    max_steps                   = 4000,
    gradient_checkpointing      = True,
    fp16                        = True,
    eval_strategy               = "steps",
    eval_steps                  = 500,
    save_strategy               = "steps",
    save_steps                  = 500,
    logging_steps               = 25,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    predict_with_generate       = True,
    generation_max_length       = 225,
    report_to                   = ["tensorboard"],
    push_to_hub                 = False,
    dataloader_num_workers      = 0,
    save_total_limit            = 3,
    optim                       = "adamw_torch",
    lr_scheduler_type           = "linear",
    label_names                 = ["labels"],
)

# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [
        EarlyStoppingCallback(early_stopping_patience=5),
        UnfreezeEncoderCallback(step=1000),
    ],
)

# ── Resume from checkpoint if exists (so we don't lose step 1-500 progress) ───
import os
checkpoints = [
    os.path.join(str(OUTPUT_DIR), d)
    for d in os.listdir(str(OUTPUT_DIR))
    if d.startswith("checkpoint-")
]
resume_from = max(checkpoints, key=os.path.getmtime) if checkpoints else None
if resume_from:
    print(f"\nResuming from checkpoint: {resume_from}")
else:
    print("\nStarting fresh training ...")

# ── Train ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Starting fine-tuning ...")
print(f"  Device       : {training_args.device}")
print(f"  FP16         : {training_args.fp16}")
print(f"  Effective BS : {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Max steps    : {training_args.max_steps}")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples  : {len(val_dataset)}")
print("=" * 60 + "\n")

trainer.train(resume_from_checkpoint=resume_from)

# ── Save ───────────────────────────────────────────────────────────────────────
final_path = OUTPUT_DIR / "final"
trainer.save_model(str(final_path))
processor.save_pretrained(str(final_path))
print(f"\n✅ Training complete. Model saved to {final_path}")
print("Next step: python q1_step3_evaluate.py")