"""
Q1 - Step 1: Download and Preprocess
FIXED: Saves clips as WAV files + manifest CSV only.
No HuggingFace Audio casting - avoids torchcodec issue on Windows.
"""

import os
import json
import re
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TARGET_SR     = 16_000
MIN_DURATION  = 1.0
MAX_DURATION  = 30.0
BASE_URL      = "https://storage.googleapis.com/upload_goai"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = [
    (967179,  825780, 443), (967179,  825727, 443),
    (1147542, 988596, 475), (1147542, 990175, 475),
    (639950,  526266, 522), (639950,  520199, 522),
    (608692,  542785, 581), (608692,  494019, 581),
    (642712,  523045, 587), (642712,  522951, 587),
    (360731,  254219, 693), (360731,  253253, 693),
    (460517,  351501, 678), (460517,  350606, 678),
    (755894,  629904, 698), (755894,  635909, 698),
    (1147324, 989901,1171), (1147324, 990783,1171),
    (347326,  240907, 559), (347326,  240909, 559),
    (378001,  270153, 584), (378001,  270150, 584),
    (592421,  475392, 438), (592421,  475356, 438),
    (362814,  255349, 780), (362814,  255381, 780),
    (902397,  767685, 853), (902397,  767869, 853),
    (1032369, 886193,1083), (1032369, 888331,1083),
    (741178,  615351, 452), (741178,  615319, 452),
    (870590,  738845,1151), (870590,  739630,1151),
    (379319,  272241,1158), (379319,  282447,1158),
    (378238,  270296, 559), (378238,  270291, 559),
    (475992,  365033, 535), (475992,  365059, 535),
    (786720,  661889,1167), (786720,  661767,1167),
    (346081,  239492, 475), (346081,  241695, 475),
    (460430,  350297, 469), (460430,  350347, 469),
    (377371,  269794, 528), (377371,  269383, 528),
    (346088,  240994, 580), (346088,  243702, 580),
    (657859,  537776, 671), (657859,  537983, 671),
    (756383,  630221, 614), (756383,  630926, 614),
    (706065,  583544, 846), (706065,  583552, 846),
    (705928,  584003,1168), (705928,  583991,1168),
    (1134557, 978393,1193), (1134557, 978484,1193),
    (756535,  629868, 442), (756535,  629862, 442),
    (557922,  443952, 795), (557922,  444282, 795),
    (411140,  302506, 468), (411140,  302503, 468),
    (772064,  645534, 458), (772064,  644742, 458),
    (427660,  330457,1193), (427660,  319431,1193),
    (1133355, 979497,1192), (1133355, 977253,1192),
    (344590,  238123, 478), (344590,  238079, 478),
    (413003,  305347,1185), (413003,  305308,1185),
    (607477,  489675, 642), (607477,  489638, 642),
    (917675,  781184, 663), (917675,  781268, 663),
    (787986,  663529, 482), (787986,  661461, 482),
    (705932,  583533, 775), (705932,  583334, 775),
    (920224,  798121, 852), (920224,  783966, 852),
    (511971,  400490,1194), (511971,  400503,1194),
    (1001923, 857737,1079), (1001923, 856801,1079),
    (409660,  301080, 494), (409660,  301057, 494),
    (477740,  367249, 601), (477740,  366972, 601),
    (377388,  269907, 836), (377388,  270037, 836),
    (427666,  319105, 926), (427666,  319126, 926),
    (887857,  754618, 589), (887857,  753435, 926),
    (1180199,1021370,1194), (1180199, 1020918,1194),
    (983900, 840793, 1146), (983900, 840781, 1146) 
]

def download_file(url, dest, timeout=30):
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  X Failed {url}: {e}")
        return False

def clean_text(text):
    if not text or text.strip() == "REDACTED":
        return ""
    text = text.replace("REDACTED", "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\u0900-\u097F\u0020-\u007E\u0964,\?\.\!\-\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Step 1: Download ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Downloading files")
print("=" * 60)
dl = {"audio_ok":0,"audio_fail":0,"trans_ok":0,"trans_fail":0}
for (folder_id, rec_id, _) in tqdm(MANIFEST, desc="Downloading"):
    ok_a = download_file(f"{BASE_URL}/{folder_id}/{rec_id}_audio.wav",
                         DATA_DIR / f"{rec_id}_audio.wav")
    ok_t = download_file(f"{BASE_URL}/{folder_id}/{rec_id}_transcription.json",
                         DATA_DIR / f"{rec_id}_transcription.json")
    if ok_a: dl["audio_ok"]   += 1
    else:    dl["audio_fail"] += 1
    if ok_t: dl["trans_ok"]   += 1
    else:    dl["trans_fail"] += 1
print(f"Download summary: {dl}")

# ── Step 2: Slice and save clips ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Slicing into utterance clips")
print("=" * 60)

CLIPS_DIR = PROCESSED_DIR / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

rows = []
skipped = {"no_audio":0,"no_trans":0,"bad_text":0,"bad_dur":0,"silence":0}
clip_idx = 0

for (folder_id, rec_id, _) in tqdm(MANIFEST, desc="Slicing"):
    audio_path = DATA_DIR / f"{rec_id}_audio.wav"
    trans_path = DATA_DIR / f"{rec_id}_transcription.json"

    if not audio_path.exists(): skipped["no_audio"] += 1; continue
    if not trans_path.exists(): skipped["no_trans"] += 1; continue

    try:
        audio_16k, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True, dtype=np.float32)
    except Exception as e:
        print(f"  X {audio_path.name}: {e}")
        skipped["no_audio"] += 1
        continue

    with open(trans_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    for seg in segments:
        text = clean_text(seg.get("text",""))
        if not text: skipped["bad_text"] += 1; continue

        dur = float(seg["end"]) - float(seg["start"])
        if dur < MIN_DURATION or dur > MAX_DURATION: skipped["bad_dur"] += 1; continue

        s = int(float(seg["start"]) * TARGET_SR)
        e = int(float(seg["end"])   * TARGET_SR)
        clip = audio_16k[s:e]

        if float(np.sqrt(np.mean(clip**2))) < 1e-4: skipped["silence"] += 1; continue

        clip_path = CLIPS_DIR / f"clip_{clip_idx:06d}.wav"
        sf.write(str(clip_path), clip, TARGET_SR, subtype="PCM_16")

        rows.append({
            "clip_path":    str(clip_path),
            "text":         text,
            "recording_id": str(rec_id),
            "duration":     round(dur, 2),
        })
        clip_idx += 1

    del audio_16k

print(f"Total clips: {len(rows)}  |  Skipped: {skipped}")
total_hrs = sum(r["duration"] for r in rows) / 3600
print(f"Total audio: {total_hrs:.2f} hrs")

# ── Step 3: Save manifests ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Saving train/val manifest CSVs")
print("=" * 60)

import random
random.seed(42)
random.shuffle(rows)

split = int(0.9 * len(rows))
train_df = pd.DataFrame(rows[:split])
val_df   = pd.DataFrame(rows[split:])

train_df.to_csv(PROCESSED_DIR / "train_manifest.csv", index=False)
val_df.to_csv(PROCESSED_DIR   / "val_manifest.csv",   index=False)

print(f"Train: {len(train_df)} clips  ({train_df['duration'].sum()/3600:.2f} hrs)")
print(f"Val:   {len(val_df)} clips    ({val_df['duration'].sum()/3600:.2f} hrs)")
print(f"\n✅ Done! Manifests saved to {PROCESSED_DIR}")
print("Now run: python q1_step2_finetune.py")