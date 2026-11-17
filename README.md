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

Train public-data experts and the autoregressive generator. Training data are specified in `data/public/training_plan.json`. ESM3 features are computed automatically. For example:

```bash
python main.py train --epochs 100 --batch-size 32 --output-dir runs/public_training
```

| Option | Default | Description |
|---|---|---|
| `--epochs N` | `100` | Epoch limit for each training stage. |
| `--batch-size N` | `32` | Training batch size. |
| `--learning-rate X` | `0.0001` | Learning rate for all stages. |

**Outputs:** `experts/` (expert weights and scaler), `interaction.pt`, `generator.pt`, `scorer.json`, training logs and a manifest.

### Iterative calibration: `calibrate`

Update the MIC head from experimental observations, retaining the frozen encoder and experts. By default this continues from the currently registered FinalModel.

```bash
python main.py calibrate --input demo/experiments.csv --output-dir runs/calibration
# Use this calibrated model for subsequent prediction
python main.py predict --model runs/calibration/model.pt --input demo/predict.fasta
```

**Input:** CSV or TSV, with one observation per row. Headers are case-sensitive. The columns are defined below:

| Column | Required | Description |
|---|---|---|
| `sequence` | Yes | Amino acid sequence. |
| `mic` | Yes | Original-scale MIC in μg/mL, e.g. `8` or `>128`. |
| `id` | No | Peptide identifier. |
| `strain` | No | Experimental strain, for traceability. |
| `round` | No | Experimental round. |

See [`demo/experiments.csv`](demo/experiments.csv) for an example input.

| Option | Default | Description |
|---|---|---|
| `--input FILE` | Required | Experimental table. |
| `--model MODEL` | `FinalModel` | `FinalModel`, `BaseModel`, or a checkpoint/configuration path. |
| `--epochs N` | `100` | Calibration epochs. |
| `--batch-size N` | `32` | Calibration batch size. |
| `--learning-rate X` | `0.0001` | Calibration learning rate. |

### Activity prediction: `predict`

Use the trained model to predict one MIC per peptide.

```bash
python main.py predict --input demo/predict.fasta --details --output-dir runs/prediction
```

**Input:** FASTA, CSV or TSV. Tables must include a `sequence` column.

```csv
id,sequence
Test1,FLRRLVKWLKKM
```

| Option | Default | Description |
|---|---|---|
| `--input FILE` | Required | Sequence file. |
| `--details` | Off | Also save expert scores in `details.csv`. |

**Output:** `predictions.csv` with `id`, `sequence` and `predicted_mic` (μg/mL). Example inputs are in [`demo/`](demo/).

### Candidate generation: `generate`

Sample the pretrained generator, filter and deduplicate sequences, optionally exclude supplied previously tested sequences with `--exclude FILE`, then rank candidates by the selected model's composite score `R_t`. No input file is required.

```bash
python main.py generate --count 100 --details --output-dir runs/candidates
```

| Option | Default | Principle and role | Accepted values | Effect of adjustment |
|---|---|---|---|---|
| `--model MODEL` | `FinalModel` | Loads the generator from the model bundle. | / | / |
| `--count N` | `100` | Returns the top N eligible candidates in descending order of `R_t`. | Integer, `N ≥ 1`, no greater than `max-proposals`. | / |
| `--pool-multiplier N` | `3` | Sets the target eligible pool to `count × N`, then selects `count` candidates. | Integer, `N ≥ 1`; sampling is limited by `max-proposals`. | Higher values broaden the selection pool and usually increase computation. |
| `--temperature X` | `1.0` | Divides the generator logits by X at each step before converting them to sampling probabilities. | `X > 0`. | Higher values flatten probabilities, usually increasing diversity and randomness, but may also yield more ineligible sequences. Lower values favor high-probability amino acids, concentrating sampling and potentially increasing repetition. `1` retains the original logit scale. |
| `--top-p X` | `0.95` | At each step, retains the smallest set of tokens whose cumulative probability, in descending order, reaches X, then samples from it. | `0 < X ≤ 1`. | Higher values allow more low-probability choices and usually increase diversity. Lower values concentrate sampling and may increase repetition. |
| `--details` | Off | Saves expert scores and the composite score `R_t` separately to `details.csv`. | / | / |

For example, `--count 100 --pool-multiplier 3` targets 300 eligible unique sequences and returns the highest-scoring 100. If the proposal limit is reached with 100–299 eligible sequences, the best 100 are still returned; fewer than 100 causes an error.

Temperature and `top-p` jointly control sampling: probabilities are first adjusted by temperature, followed by `top-p` truncation.

**Outputs:** `candidates.csv` (`rank`, `id`, `sequence`, `predicted_mic` in μg/mL), `candidates.fasta` and `run.json`.

For NVIDIA GPUs, install a driver-compatible CUDA build in the Conda environment using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then run:

```bash
python main.py predict --input demo/predict.fasta --device cuda
python main.py calibrate --input demo/experiments.csv --device cuda
python main.py generate --count 100 --device cuda
python main.py train --device cuda
```
