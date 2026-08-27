# MDRAMP

Mechanism-aware closed-loop modeling for multidrug-resistant antimicrobial peptide discovery.

MDRAMP is an early research implementation of a hierarchical MDR-AMP discovery workflow. The project combines strict ESM3 peptide/protein representations, public AMP initialization, private-round state updates, membrane-mechanism evidence, peptide-target interaction evidence, quantitative MIC prioritization, and a gated consensus scoring function.

```text
public AMP data -> frozen Expert backbone + MIC head -> M_R0
private/R0 labels -> next-state MIC update -> M_R1
private/R1 labels -> next-state MIC update -> M_R2
...
model state -> p_cons + A_t -> R_t candidate ranking
```

This repository is intended to be developed and released as an open-source implementation of the model workflow. It does not include alternative sequence encoders: ESM3 is the required representation model.

## Overview

MDRAMP follows a multi-stage modeling strategy:

1. Public AMP records from APD6, CAMP, DBAASP, dbAMP, and DRAMP are parsed, cleaned, deduplicated, and split into a public source-domain training set.
2. ESM3 is used to generate peptide and target protein embeddings.
3. A public-initialized model state, `M_R0`, is trained from public AMP data and teacher-derived membrane evidence.
4. Released private round data are visible only to the subsequent state. In the paper-faithful path, R0-R3 update only the lightweight MIC head; ESM3, Expert ensembles and the consistency integrator remain frozen.
5. Candidates receive the stable biological consistency score `p_cons`, MIC-derived activity priority `A_t`, and deployed ranking score `R_t = p_cons * A_t`.

## Key Features

- Strict ESM3 peptide and target protein embeddings.
- Parsers for public raw AMP data, private round data, and target FASTA files.
- Multi-round fine-tuning interface for closed-loop model states.
- Required wrappers for external teacher models:
  - `otherModels/Pore-Forming`
  - `otherModels/TPepPro`
- Three expert streams:
  - AMP Prior Expert
  - Membrane Mechanism Expert
  - Target Affinity Expert
- Noisy-OR aggregation for peptide-target affinity evidence.
- Gated consensus scoring for mechanism-aware candidate ranking.
- Five-member ensemble and uncertainty-aware scoring interfaces.
- Censor-aware log2(MIC) regression head and `R_t` deployment ranking.
- State manifests and prequential label-isolation guards.
- Candidate guardrails, mixed selection, Pareto selection and Pep-B optimization interfaces.
- Post-campaign evidential calibration utilities.
- Project-local outputs for processed data, embeddings, teacher scores, checkpoints, and final scores.

## Repository Layout

```text
MDRAMP_GitHub_release/
  configs/       YAML configuration files
  data/          raw public data, private round data, and target proteins
  otherModels/   required external teacher models
  outputs/       generated artifacts
  src/           source code
  README.md
```

Source modules:

```text
src/data/       data parsing, sequence cleaning, labels, and split utilities
src/features/   ESM3 embedding generation and physicochemical descriptors
src/external/   Pore-Forming and TPepPro adapters
src/models/     expert modules, gated integrator, and system wrapper
src/train/      public pretraining, MIC-head training, legacy and round training interfaces
src/score/      target-affinity prediction, deployed scoring, calibration and contribution analysis
src/loop/       model-state manifests and prequential leakage guards
src/selection/  candidate guardrails, mixed policy and Pareto selection
src/optimization/ Pep-B directed optimization utilities
src/utils/      FASTA, schema, hashing, and checkpoint helpers
```

## Data

Expected input structure:

```text
data/rawdata/
  ADP6/
  CAMP/
  DBAASP/
  dbAMP/
  DRAMP/

data/private/R0.txt
data/target/target.fa
```

`data/private/R0.txt` is treated as first-round released private fine-tuning data. It is not treated as the final candidate pool and must not be used for same-round candidate selection, calibration, threshold tuning, or replacement.

## Model Formulation

For a peptide sequence `x`, MDRAMP estimates three evidence streams:

```text
S_prior(x)  AMP-like sequence feasibility
S_mem(x)    membrane-mechanism evidence
S_aff(x)    aggregated target-interaction evidence
```

Pair-level peptide-target affinity scores are aggregated across targets using Noisy-OR:

```text
S_aff(x) = 1 - product_t(1 - s_aff(x,t))
```

The final gated consensus score is:

```text
g_prior(x) = sigmoid(gamma * (S_prior(x) - tau))
E_comp(x) = 1 - (1 - S_mem(x)) * (1 - S_aff(x))
E_syn(x) = S_mem(x) * S_aff(x)
E_mech(x) = (1 - lambda_syn) * E_comp(x) + lambda_syn * E_syn(x)
p_cons(x) = g_prior(x) * E_mech(x)
A_t(x) = sigmoid(5 - predicted_log2_MIC(x))
R_t(x) = p_cons(x) * A_t(x)
```

`p_cons(x)` is the stable biological consistency-priority component, not an absolute probability. `A_t(x)` is the state-specific MIC-derived activity priority. `R_t(x)` is the prospective deployed ranking score. Post-campaign evidential probabilities are a separate calibration analysis and must not be used as the historical R0-R5 selection score.

## Installation

Python 3.12 is recommended for the ESM3 environment.

Install ESM3 from the official Biohub repository:

```bash
pip install esm@git+https://github.com/Biohub/esm.git@main
```

Install common runtime dependencies:

```bash
pip install numpy pandas torch scikit-learn biopython pyyaml
```

Run the dependency-light checks from the repository root:

```bash
pytest -q tests
python scripts/dry_run_foundation.py
```

These checks do not require ESM3 or external teacher weights. Formal embedding and teacher inference require the external environments described below.

Some components may require additional dependencies from ESM3, Pore-Forming, or TPepPro. For ESM3 checkpoint caching, use a project-local cache:

```bash
export HF_HOME=outputs/hf_cache
export HUGGINGFACE_HUB_CACHE=outputs/hf_cache/hub
```

## Quick Start

Run all commands from the repository root.

### 1. Process Data

```bash
python -m src.data.build_processed --root . --round R0
```

Main outputs:

```text
outputs/processed/public_cluster_split.csv
outputs/processed/private_R0.csv
outputs/processed/targets.csv
outputs/processed/processed_metadata.json
```

### 2. Generate ESM3 Embeddings

Run these commands on a machine configured for ESM3:

```bash
python -m src.features.build_esm3_embeddings --dataset public --device cuda --hf-home outputs/hf_cache
python -m src.features.build_esm3_embeddings --dataset private_R0 --device cuda --hf-home outputs/hf_cache
python -m src.features.build_esm3_embeddings --dataset targets --device cuda --hf-home outputs/hf_cache
```

Main outputs:

```text
outputs/esm3_embeddings/public_peptides.npz
outputs/esm3_embeddings/public_peptides_metadata.csv
outputs/esm3_embeddings/private_R0_peptides.npz
outputs/esm3_embeddings/private_R0_peptides_metadata.csv
outputs/esm3_embeddings/targets.npz
outputs/esm3_embeddings/targets_metadata.csv
```

### 3. Build Peptide Features

```bash
python -m src.features.physicochemical \
  --input outputs/processed/public_cluster_split.csv \
  --output outputs/processed/public_features.csv

python -m src.features.physicochemical \
  --input outputs/processed/private_R0.csv \
  --output outputs/processed/private_R0_features.csv
```

### 4. Generate Pore-Forming Teacher Scores

```bash
python -m src.external.pore_forming_wrapper \
  --input outputs/processed/public_cluster_split.csv \
  --output outputs/teacher_scores/pore_public.csv \
  --id-column sequence \
  --sequence-column sequence

python -m src.external.pore_forming_wrapper \
  --input outputs/processed/private_R0.csv \
  --output outputs/teacher_scores/pore_private_R0.csv \
  --id-column private_record_id \
  --sequence-column sequence
```

### 5. Generate TPepPro Pair Scores

Create peptide-target templates:

```bash
python -m src.external.tpeppro_wrapper template \
  --peptides outputs/processed/private_R0.csv \
  --targets outputs/processed/targets.csv \
  --output outputs/teacher_scores/tpeppro_private_R0_template.tsv
```

Run TPepPro in its required environment, then convert and aggregate predictions:

```bash
python -m src.external.convert_tpeppro_output \
  --raw outputs/teacher_scores/tpeppro_private_R0_raw_predictions.csv \
  --output outputs/teacher_scores/tpeppro_private_R0_pair_scores.csv

python -m src.external.tpeppro_wrapper aggregate \
  --pair-scores outputs/teacher_scores/tpeppro_private_R0_pair_scores.csv \
  --aggregation-coefficient '<frozen manuscript experiment value>' \
  --output outputs/teacher_scores/tpeppro_private_R0_aggregated.csv
```

### 6. Train Public-Initialized `M_R0`

```bash
python -m src.train.pretrain_public \
  --positive-features outputs/processed/public_features.csv \
  --positive-embeddings outputs/esm3_embeddings/public_peptides.npz \
  --positive-embedding-metadata outputs/esm3_embeddings/public_peptides_metadata.csv \
  --background-features outputs/processed/swissprot_background_features.csv \
  --background-embeddings outputs/esm3_embeddings/swissprot_background.npz \
  --background-embedding-metadata outputs/esm3_embeddings/swissprot_background_metadata.csv \
  --positive-prior '<frozen experiment value>' \
  --manifold-loss-weight '<frozen experiment value>' \
  --amp-branch-weight '<frozen experiment value>' \
  --pore-scores outputs/teacher_scores/pore_public.csv \
  --checkpoint-dir outputs/checkpoints/M_R0
```

This command is strict: the AMP positive and operational-unlabelled sets must be
1:1 length-matched sets. It writes five-member AMP and membrane ensembles plus
the frozen state feature scaler. Missing membrane auxiliary annotations are
masked and never converted to negative labels.
The three AMP values above are required because the Word methods name these
terms but do not report their frozen numeric values; no defaults are fabricated.

Train the target-affinity expert from pair-level teacher scores:

```bash
python -m src.train.train_target_affinity \
  --pair-scores outputs/teacher_scores/tpeppro_private_R0_pair_scores.csv \
  --peptide-features outputs/processed/private_R0_features.csv \
  --peptide-embeddings outputs/esm3_embeddings/private_R0_peptides.npz \
  --peptide-embedding-metadata outputs/esm3_embeddings/private_R0_peptides_metadata.csv \
  --target-embeddings outputs/esm3_embeddings/targets.npz \
  --target-embedding-metadata outputs/esm3_embeddings/targets_metadata.csv \
  --peptide-id-column private_record_id \
  --scaler outputs/checkpoints/M_R0/feature_scaler.json \
  --output outputs/checkpoints/M_R0/target_affinity_ensemble.pt
```

The pair table must contain equal numbers of `label_type=positive` and
`label_type=background` rows. The trained ensemble uses the independent
256-dimensional projections and rank-64 bilinear interaction described in the
manuscript.

### 7. Update the MIC head for `M_R0 -> M_R1`

The manuscript-faithful R0-R3 update path trains only the lightweight quantitative MIC head. The old membrane Expert fine-tuning command is retained only as an explicitly marked early-prototype path and is not the paper workflow.

```bash
python -m src.train.train_mic_head \
  --mic-ledger outputs/processed/mic_R0.csv \
  --peptide-features outputs/processed/private_R0_features.csv \
  --peptide-embeddings outputs/esm3_embeddings/private_R0_peptides.npz \
  --peptide-embedding-metadata outputs/esm3_embeddings/private_R0_peptides_metadata.csv \
  --expert-means outputs/scores/private_R0_expert_means.csv \
  --state-manifest outputs/checkpoints/M_R1/state_manifest.json \
  --current-round R1 \
  --scaler outputs/checkpoints/M_R0/feature_scaler.json \
  --peptide-id-column private_record_id \
  --output outputs/checkpoints/M_R1/mic_head.pt
```

For later rounds, replace the private round input and checkpoint paths according to the model-state transition:

```text
private/Rk -> M_R(k+1)
```

### 8. Score Candidates

Predict target-affinity evidence:

```bash
python -m src.score.predict_target_affinity \
  --checkpoint outputs/checkpoints/M_R0/target_affinity_ensemble.pt \
  --scaler outputs/checkpoints/M_R0/feature_scaler.json \
  --peptide-features outputs/processed/private_R0_features.csv \
  --peptide-embeddings outputs/esm3_embeddings/private_R0_peptides.npz \
  --peptide-embedding-metadata outputs/esm3_embeddings/private_R0_peptides_metadata.csv \
  --target-embeddings outputs/esm3_embeddings/targets.npz \
  --target-embedding-metadata outputs/esm3_embeddings/targets_metadata.csv \
  --peptide-id-column private_record_id \
  --aggregation-coefficient '<frozen manuscript experiment value>' \
  --aggregate-output outputs/teacher_scores/model_private_R0_aggregated.csv
```

The Word methods state that the top-five attention and Noisy-OR terms use a
prespecified coefficient but do not report its numeric value. The CLI therefore
requires the archived experiment value explicitly and does not invent a default.

Compute final gated consensus scores:

```bash
python -m src.score.score_candidates \
  --candidates outputs/processed/private_R0.csv \
  --embeddings outputs/esm3_embeddings/private_R0_peptides.npz \
  --embedding-metadata outputs/esm3_embeddings/private_R0_peptides_metadata.csv \
  --id-column private_record_id \
  --scaler outputs/checkpoints/M_R0/feature_scaler.json \
  --amp-prior outputs/checkpoints/M_R0/amp_prior_ensemble.pt \
  --membrane outputs/checkpoints/M_R0/membrane_ensemble.pt \
  --target-affinity outputs/teacher_scores/model_private_R0_aggregated.csv \
  --mic-checkpoint outputs/checkpoints/M_R1/mic_head.pt \
  --output outputs/scores/candidate_scores.csv
```

### 9. Analyze Evidence Contributions

```bash
python -m src.score.contribution_analysis \
  --scores outputs/scores/candidate_scores.csv \
  --output outputs/scores/contribution_analysis.csv
```

## Output Schemas

TPepPro pair scores:

```text
peptide_id,target_id,s_aff_pair
```

Aggregated target-affinity scores:

```text
peptide_id,member_0,...,member_4,S_int_mean,S_int_std,top_target_id,top_target_score
```

Candidate scores:

```text
rank
candidate id
sequence
S_prior
S_mem
S_int
mu_prior,sigma_prior
mu_mem,sigma_mem
mu_int,sigma_int
g_prior
E_comp
E_syn
E_mech
p_cons
predicted_log2_mic
A_t
R_t
top_target_id
top_target_score
```

## Reproducibility Notes

- ESM3 embeddings should be generated with the same ESM3 checkpoint and device settings for a given experiment.
- `outputs/processed/processed_metadata.json` records processed data counts and provenance.
- The in-repository public split uses deterministic global alignment identity at the fixed 90% threshold, including insertion/deletion gaps.
- TPepPro may require a separate legacy environment. MDRAMP consumes converted pair-level TPepPro predictions through a stable CSV schema.
- Pore-Forming may require its original tokenizer, transformer and checkpoint environment. Import failures are reported explicitly; no substitute teacher is silently used.
- HemoPI-2 is required for the R5 haemolysis-aware path. `src.selection.r5_safety` invokes a pinned external HemoPI-2 command after antibacterial scoring and rejects missing, failed or out-of-range risk outputs.
- The historical prospective ledger, fixed challenge set and archived checkpoints are not included in this early repository snapshot; manuscript-level result replay is blocked until those artifacts are supplied.
- `outputs/` contains generated artifacts and can be excluded from version control except for small release examples.

## Closed-Loop Usage Rule

Private round `Rk` can only update the next model state, `M_R(k+1)`.

It must not be used for same-round:

- candidate ranking
- threshold adjustment
- candidate replacement
- calibration
- hyperparameter selection

## Replication Status and Development Plan

The paper-to-code implementation plan is maintained in [`docs/MDRAMP_PAPER_REPLICATION_PLAN_ZH.md`](docs/MDRAMP_PAPER_REPLICATION_PLAN_ZH.md). The active training and scoring entrypoints now enforce the paper Expert architectures, five-member ensembles, uncertainty adjustment, frozen scaler, prequential MIC update and complete `R_t` ranking. Exact numerical replay of the reported campaign remains blocked until the pinned external model environments and weights, archived 21,445-protein snapshot/panel, historical state checkpoints, experimental ledger and fixed challenge set are supplied.

Run the local checks before attempting external inference:

```bash
pytest -q tests
python scripts/dry_run_foundation.py
```

The command `python -m src.train.finetune_round` is intentionally blocked by default because the manuscript does not update Expert weights during R0-R3. Use `src.train.train_mic_head` for the paper-faithful MIC-head update. Pass `--allow-legacy-expert-finetune` only when explicitly reproducing the early prototype behavior.

## Citation

If you use this repository, please cite the associated manuscript or preprint when available.

## License

Add the final open-source license before public release.
