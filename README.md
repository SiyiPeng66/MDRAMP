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

The default workflow uses CPU and Python 3.12. Tested on macOS; Linux is supported but has not been independently tested. On Windows, use WSL2.

Models download automatically on first use from [MDRAMP](https://huggingface.co/BillyLiang/MDRAMP) and [ESM3](https://huggingface.co/biohub/esm3-sm-open-v1). The complete assets occupy approximately 5.75 GiB. Subsequent runs reuse local files and can work offline.

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

Outputs are saved to a new directory under `runs/`. Use `--output-dir DIR` to choose a destination and `--help` for command options. Public training automatically computes missing sequence features; full CPU encoding and training can take substantial time.

## Models

| Model | Use |
|---|---|
| **BaseModel** | Public-data-trained experts and generator. Requires experimental calibration before MIC prediction. |
| **FinalModel** | Calibrated model; default for prediction, further calibration and candidate scoring. |

Use `--model BaseModel` or `--model path/to/model.pt` to select another starting model. Custom checkpoints require a companion `scorer.json`. Calibration saves a new model; add `--update-final` to make it the default.

## Input and output

**Prediction:** FASTA, CSV, TSV or XLSX (first worksheet). Tables require `sequence`; `id` is optional. Use the 20 standard amino acids. Strain information is not required.

**Calibration:** CSV, TSV or XLSX, one observation per row:

| Column | Required | Value |
|---|---|---|
| `sequence` | Yes | Amino acid sequence |
| `mic` | Yes | Input MIC in μg/mL, e.g. `8` or `>128` |
| `id` | No | Peptide identifier |
| `strain` | No | Experimental strain |
| `round` | No | Experimental round |

The default calibration target is the best MIC across strains **divided by 10**. Do not pre-divide input values; use `--mic-divisor 1` for unscaled labels. See [model and data notes](docs/MODELS_AND_DATA.md) for label provenance and interpretation.

| Command | Main output |
|---|---|
| `predict` | `predictions.csv`: `id`, `sequence`, `predicted_mic` |
| `calibrate` | `model.pt`, `scorer.json`, `calibration.csv` |
| `generate` | `candidates.csv`, `candidates.fasta` |
| `train` | Expert weights, generator and `scorer.json` |

Examples are in [`demo/`](demo/README.md); an empty calibration template is provided at [`templates/experiments.csv`](templates/experiments.csv). Training options include `--epochs`, `--batch-size`, `--learning-rate` and `--seed`. Add `--details` to prediction or generation for expert scores.

## Model and data notes

FinalModel was fitted with adjusted MIC labels and ranking preferences. Its outputs are model estimates, not new experimental measurements. Candidate generation samples the pretrained autoregressive generator, applies filters and ranks with the selected model; it does not retrain the generator per request. The demo peptides participated in calibration and are not an independent test set.

See [model and data notes](docs/MODELS_AND_DATA.md) for asset versions, training inputs, candidate tables and reproducibility details. Upstream models and datasets retain their respective licenses.

## Tests

```bash
python -m pytest -q
```
