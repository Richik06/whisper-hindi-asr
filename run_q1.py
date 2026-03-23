"""
Q1 - Master Runner
==================
Run this single script to execute the full pipeline end-to-end.
Or run each step independently.

Usage:
    python run_q1.py

Steps:
    1. Download + preprocess data
    2. Fine-tune Whisper-small
    3. Evaluate on FLEURS Hindi test set
    4. Error analysis, taxonomy, fixes
"""

import subprocess
import sys

steps = [
    ("STEP 1: Download & Preprocess", "q1_step1_download_preprocess.py"),
    ("STEP 2: Fine-tune Whisper-small",  "q1_step2_finetune.py"),
    ("STEP 3: Evaluate on FLEURS",       "q1_step3_evaluate.py"),
    ("STEP 4: Error Analysis & Fixes",   "q1_step4_error_analysis.py"),
]

for title, script in steps:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], check=True)
    if result.returncode != 0:
        print(f"❌ {script} failed. Stopping.")
        sys.exit(1)

print("\n✅ All steps complete.")