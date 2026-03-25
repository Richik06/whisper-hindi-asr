# Hindi ASR Pipeline — Josh Talks AI Researcher Intern Assignment

Complete solution for all 4 questions: Hindi ASR fine-tuning, ASR post-processing pipeline, Hindi spell checking, and lattice-based WER evaluation.

---

## Tech Stack

| Component | Detail |
|---|---|
| Model | OpenAI Whisper-small |
| GPU | NVIDIA RTX 4050 8GB |
| Python | 3.11 |
| Key Libraries | HuggingFace Transformers, PyTorch, librosa, soundfile, evaluate |

---

## Setup

```bash
# Step 1: Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Step 2: Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Step 3: Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 4: Install remaining dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
whisper_hindi/
│
├── q1_step1_download_preprocess.py      # Download audio + transcriptions, preprocess
├── q1_step2_finetune.py                 # Fine-tune Whisper-small on Hindi data
├── q1_step3_evaluate.py                 # Evaluate baseline vs fine-tuned model
├── q1_step4_error_analysis.py           # Error taxonomy, sampling, fixes
├── add_reasoning_to_taxonomy.py         # Add reasoning column to error CSV
├── run_q1.py                            # Master runner for Q1 (runs all 4 steps)
│
├── q2_pipeline.py                       # Number normalization + English detection
│
├── q3_spell_check.py                    # Hindi spell checker on ~1,77,000 words
│
├── q4_lattice_wer.py                    # Lattice-based WER evaluation
│
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   ├── raw/                             # Downloaded .wav + .json files [gitignored]
│   └── processed/
│       ├── train_manifest.csv           # Train split (4266 clips)
│       ├── val_manifest.csv             # Val split (475 clips)
│       └── clips/                       # Utterance-level .wav files [gitignored]
│
├── results/
│   ├── wer_table.csv                    # Q1: WER comparison table
│   ├── per_utterance_predictions.csv    # Q1: All per-utterance predictions
│   ├── error_taxonomy_sample.csv        # Q1: 25+ error samples with reasoning
│   ├── fix_roman_before_after.csv       # Q1: Before/after Fix #1
│   │
│   ├── q2/
│   │   ├── q2_pipeline_output.csv           # Q2: 200-row pipeline output with reasoning
│   │   └── q2_real_examples_with_reasoning.csv  # Q2: Real data examples
│   │
│   ├── q3/
│   │   ├── q3_word_classifications.csv      # Q3: All words → label + confidence + reason
│   │   ├── q3_low_confidence_review.csv     # Q3: 50-word manual review
│   │   └── q3_summary.csv                   # Q3: Summary statistics
│   │
│   └── q4/
│       ├── q4_wer_results.csv               # Q4: Per-segment per-model WER
│       ├── q4_model_summary.csv             # Q4: Avg standard vs lattice WER per model
│       └── q4_lattice_corrections.csv       # Q4: Positions where lattice corrected ref
│
├── models/                              # Fine-tuned model weights [gitignored]
└── logs/                                # TensorBoard training logs [gitignored]
```

---

## Running the Code

### Q1 — Whisper Fine-tuning

```bash
# Run all steps at once
python run_q1.py

# Or step by step
python q1_step1_download_preprocess.py   # ~20-30 min (download + preprocess)
python q1_step2_finetune.py              # ~3-5 hours on RTX 4050
python q1_step3_evaluate.py             # ~20-30 min (evaluation)
python q1_step4_error_analysis.py       # ~2-3 min (error analysis)
python add_reasoning_to_taxonomy.py     # ~1 min (adds reasoning column)
```

### Q2 — ASR Post-processing Pipeline

```bash
python q2_pipeline.py    # ~15-20 min (runs baseline Whisper on 200 clips)
```

### Q3 — Hindi Spell Checker

```bash
python q3_spell_check.py    # ~2-3 min (downloads word list, classifies all words)
```

### Q4 — Lattice WER Evaluation

```bash
python q4_lattice_wer.py    # ~1 min (downloads sheet data, computes WER)
```

---

## Data Sources

| Data | URL Pattern |
|---|---|
| Audio files | `https://storage.googleapis.com/upload_goai/{folder_id}/{rec_id}_audio.wav` |
| Transcriptions | `https://storage.googleapis.com/upload_goai/{folder_id}/{rec_id}_transcription.json` |
| Metadata | `https://storage.googleapis.com/upload_goai/{folder_id}/{rec_id}_metadata.json` |
| Dataset manifest | [Google Sheet](https://docs.google.com/spreadsheets/d/1bujiO2NgtHlgqPlNvYAQf5_7ZcXARlIfNX5HNb9f8cI) |
| Q3 word list | [Google Sheet](https://docs.google.com/spreadsheets/d/17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw) |
| Q4 transcriptions | [Google Sheet](https://docs.google.com/spreadsheets/d/1J_I0raoRNbe29HiAPD5FROTr0jC93YtSkjOrIglKEjU) |

---

## Key Results

### Q1 — WER Comparison
| Model | WER (%) | CER (%) |
|---|---|---|
| whisper-small baseline (zero-shot) | 134.05 | 95.95 |
| whisper-small fine-tuned (~10h) | **37.67** | **18.77** |

### Q2 — Pipeline Summary (200 utterances)
| Operation | Utterances Changed | Helped WER | Hurt WER |
|---|---|---|---|
| Number normalization | ~15 | ~8 | ~3 |
| English word tagging | ~7 | — | — |

### Q3 — Spell Check Summary
| Category | Count |
|---|---|
| Total unique words | ~1,77,000 |
| Correctly spelled | ~1,60,000+ |
| Incorrectly spelled | ~17,000 |
| Low confidence accuracy | ~65–75% |

### Q4 — Lattice WER (6 models, 46 segments)
Lattice-based evaluation reduces WER for models unfairly penalized by rigid reference (punctuation differences, valid spelling variants, filler word omissions). Models with genuinely correct output remain unchanged.

---

## Notes

- **Audio files** are not committed (too large). Re-run `q1_step1_download_preprocess.py` to download.
- **Model weights** are not committed. Re-run `q1_step2_finetune.py` to reproduce.
- All results CSVs are committed and open correctly in Excel (saved as UTF-8 with BOM).
- Hindi text displays correctly in VSCode. For Excel, use **Data → From Text/CSV → UTF-8**.