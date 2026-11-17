# MDRAMP

[English](README.md) | [简体中文](README-Zh.md)

Antimicrobial peptide modeling with public-data pretraining, iterative calibration, MIC prediction and candidate generation.

## Installation

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Conda, then run from the repository root:

```bash
conda env create -f environment.yml
conda activate mdramp
python main.py --help
```

Runtime: Python 3.12. CPU is the default; use `--device cuda` for an NVIDIA GPU.

Models download automatically on first use from [MDRAMP](https://huggingface.co/BillyLiang/MDRAMP) and [ESM3](https://huggingface.co/biohub/esm3-sm-open-v1).

## Quick start

```bash
# Predict peptide MIC
python main.py predict --input demo/predict.fasta

# Calibrate the current model with experimental data
python main.py calibrate --input demo/experiments.csv

# Generate 100 candidates
python main.py generate --count 100

# Train from the included public data
python main.py train
```

Outputs are saved to a new directory under `runs/`. Use `--output-dir DIR` to choose a destination and `--help` for command options.

## Models

| Model | Use |
|---|---|
| **BaseModel** | Public-data-trained experts and generator. Requires experimental calibration before MIC prediction. |
| **FinalModel** | Calibrated model; default for prediction, further calibration and candidate scoring. |

Use `--model BaseModel` or `--model path/to/model.pt` to select another starting model. Custom checkpoints require a companion `scorer.json`. Calibration saves a new model; add `--update-final` to make it the default.

## Command reference

All commands accept the following options:

| Option | Default | Description |
|---|---|---|
| `--output-dir DIR` | New timestamped directory in `runs/` | Save results; the directory must not already exist. |
| `--device {cpu,cuda}` | `cpu` | Compute device; CUDA requires an NVIDIA driver and CUDA-enabled PyTorch. |
| `--threads N` | `2` | CPU thread count. |
| `--seed N` | `11` | Random seed; public expert ensembles use the fixed seeds in the training plan. |
| `--help` | — | Show command-specific help. |

### Training: `train`

Train public-data experts and the autoregressive generator. Training data are specified in `data/public/training_plan.json`. ESM3 features are computed automatically.

```bash
python main.py train --epochs 100 --batch-size 32 --output-dir runs/public_training
```

| Option | Default | Description |
|---|---|---|
| `--epochs N` | `100` | Epoch limit for each training stage. |
| `--batch-size N` | `32` | Training batch size. |
| `--learning-rate X` | `0.0001` | Learning rate for all stages. |

**Outputs:** `experts/` (expert weights and scaler), `interaction.pt`, `generator.pt`, `scorer.json`, training logs and a manifest. The new bundle has no MIC head and does not replace the default model. Initialize its MIC head with:

```bash
python main.py calibrate --model runs/public_training/scorer.json --input demo/experiments.csv
```

### Iterative calibration: `calibrate`

Update the MIC head from experimental observations, retaining the frozen encoder and experts. By default this continues from the currently registered FinalModel.

```bash
python main.py calibrate --input demo/experiments.csv --output-dir runs/calibration
# Use this calibrated model for subsequent prediction
python main.py predict --model runs/calibration/model.pt --input demo/predict.fasta
```

**Input:** CSV, TSV or XLSX (first worksheet), one observation per row. Headers are case-sensitive.

| Column | Required | Description |
|---|---|---|
| `sequence` | Yes | Amino acid sequence. |
| `mic` | Yes | Original-scale MIC in μg/mL, e.g. `8` or `>128`. |
| `id` | No | Peptide identifier. |
| `strain` | No | Experimental strain, for traceability. |
| `round` | No | Experimental round. |

See [`demo/experiments.csv`](demo/experiments.csv) for an example input. The best MIC interval across strains is used for each sequence. The default target is **divided by 10**; do not pre-divide inputs. Use `--mic-divisor 1` for unscaled labels.

| Option | Default | Description |
|---|---|---|
| `--input FILE` | Required | Experimental table. |
| `--model MODEL` | `FinalModel` | `FinalModel`, `BaseModel`, or a checkpoint/configuration path. |
| `--epochs N` | `100` | Calibration epochs. |
| `--batch-size N` | `32` | Calibration batch size. |
| `--learning-rate X` | `0.0001` | Calibration learning rate. |
| `--mic-divisor X` | `10` | Positive divisor applied to training MIC labels. |
| `--update-final` | Off | Register the successfully calibrated model as the default FinalModel. |

**Outputs:** `model.pt`, `scorer.json`, `calibration.csv` (`id`, `sequence`, `before_mic`, `after_mic`), processed observations/targets and training provenance. `before_mic` is empty when initializing from BaseModel. Preserve referenced shared assets when moving a calibrated bundle.

### Activity prediction: `predict`

Predict one MIC per peptide using FinalModel or a calibrated custom model. BaseModel must be calibrated first.

```bash
python main.py predict --input demo/predict.fasta --details --output-dir runs/prediction
```

**Input:** FASTA, CSV, TSV or XLSX (first worksheet). Tables require `sequence`; `id` is optional. FASTA titles supply identifiers. Only the 20 standard amino acids are accepted; whitespace is removed and letters are uppercased. Strain information is not required.

```csv
id,sequence
Test1,FLRRLVKWLKKM
```

| Option | Default | Description |
|---|---|---|
| `--input FILE` | Required | Sequence file. |
| `--model MODEL` | `FinalModel` | Registered model or calibrated checkpoint/configuration path. |
| `--details` | Off | Also save expert scores in `details.csv`. |

**Outputs:** `predictions.csv` with `id`, `sequence`, `predicted_mic` (μg/mL), and `run.json` with parameters and asset hashes. Example inputs are in [`demo/`](demo/).

### Candidate generation: `generate`

Sample the pretrained generator, filter and deduplicate sequences, optionally exclude supplied previously tested sequences with `--exclude FILE`, then rank candidates by the selected model's composite score `R_t`. No input file is required.

```bash
python main.py generate --count 100 --details --output-dir runs/candidates
```

| Option | Default | Principle and role | Accepted values | Effect of adjustment |
|---|---|---|---|---|
| `--model MODEL` | `FinalModel` | Loads the bundle's generator and ranks candidates using its MIC head and composite score `R_t`. | `FinalModel` or a calibrated `.pt`/`scorer.json` path; uncalibrated BaseModel is not accepted. | Changing bundles can change sampling, MIC and ranks; changing only the MIC head primarily changes scoring and selection. |
| `--count N` | `100` | Returns the top N eligible candidates ranked by `R_t`. | Integer, `N ≥ 1` and `N ≤ max-proposals`. | Higher values return more candidates and proportionally enlarge the target pool, usually increasing runtime; they do not imply higher mean activity. |
| `--pool-multiplier N` | `3` | Sets the target eligible pool to `count × N` before selecting `count` candidates. | Integer, `N ≥ 1`, subject to `max-proposals`. | Higher values broaden the selection pool and usually increase computation; scores need not improve monotonically. Sampling stops at the target, with possible batch overshoot. |
| `--max-proposals N` | `20000` | Caps the number of sequences sampled before filtering and deduplication. | Integer, `N ≥ count`. | Higher values allow more attempts when duplicates or rejected sequences are common, potentially increasing runtime. The full budget is not used once the target pool is reached. |
| `--temperature X` | `1.0` | Divides next-token logits by X before converting them to probabilities. | Finite real number, `X > 0`. | Higher values flatten probabilities, usually increasing diversity and randomness but potentially rejection rates. Lower values favor high-probability residues and can increase repetition. `1` retains the original logit scale. |
| `--top-p X` | `0.95` | Retains the smallest descending-probability token set whose cumulative probability reaches X, then samples from it. | Real number, `0 < X ≤ 1`. | Higher values admit more low-probability choices and usually increase diversity. Lower values concentrate sampling and can increase repetition. `1` disables probability-tail truncation; length and token constraints still apply. |
| `--exclude FILE` | Unset | Excludes exact sequences listed in the input file. | FASTA, CSV, TSV or XLSX; tables require `sequence`. | More exclusions reduce the eligible pool and may require a larger sampling budget; similar sequences are not automatically excluded. |
| `--details` | Off | Saves expert scores and `R_t` to a separate `details.csv`. | Flag; no value required. | Adds output fields without changing sampling, MIC predictions or ranks. |

For example, `--count 100 --pool-multiplier 3` targets 300 eligible unique sequences and returns the highest-scoring 100. If the proposal limit is reached with 100–299 eligible sequences, the best 100 are still returned; fewer than 100 causes an error.

Temperature is applied before top-p truncation. Both control sampling diversity; greater diversity does not establish greater antimicrobial activity, which still requires scoring and experimental confirmation.

**Outputs:** `candidates.csv` (`rank`, `id`, `sequence`, `predicted_mic` in μg/mL), `candidates.fasta` and `run.json`. Increase `--max-proposals` if too few candidates pass the filters.

For NVIDIA GPUs, install a driver-compatible CUDA build in the Conda environment using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then run:

```bash
python main.py predict --input demo/predict.fasta --device cuda
python main.py calibrate --input demo/experiments.csv --device cuda
python main.py generate --count 100 --device cuda
python main.py train --device cuda
```
