# MDRAMP

Mechanism-aware closed-loop modeling for multidrug-resistant antimicrobial peptide discovery.

MDRAMP is a research codebase for reproducing the modeling logic of a hierarchical MDR-AMP discovery workflow. The project combines strict ESM3 peptide/protein representations, public AMP pretraining, private round fine-tuning, membrane-mechanism evidence, peptide-target interaction evidence, and a gated consensus scoring function for candidate prioritization.

```text
public AMP data -> M_R0
private/R0 -> M_R1
private/R1 -> M_R2
...
model state -> candidate scoring -> p_cons ranking
```

This repository is intended to be developed and released as an open-source implementation of the model workflow. It does not include alternative sequence encoders: ESM3 is the required representation model.

## Overview

MDRAMP follows a multi-stage modeling strategy:

1. Public AMP records from APD6, CAMP, DBAASP, dbAMP, and DRAMP are parsed, cleaned, deduplicated, and split into a public source-domain training set.
2. ESM3 is used to generate peptide and target protein embeddings.
3. A public-initialized model state, `M_R0`, is trained from public AMP data and teacher-derived membrane evidence.
4. Released private round data, such as `private/R0.txt`, are used only to fine-tune the next model state, for example `M_R0 -> M_R1`.
5. Candidates are scored by integrating AMP prior evidence, membrane-mechanism evidence, and target-affinity evidence through a manuscript-faithful gated consensus function.

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
src/train/      public pretraining, private round fine-tuning, target-affinity training
src/score/      target-affinity prediction, candidate scoring, contribution analysis
src/loop/       model-state metadata
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
```

`p_cons(x)` is used for ranking and model-state comparison. It should not be interpreted as an absolute probability of antimicrobial activity.

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
  --output outputs/teacher_scores/tpeppro_private_R0_aggregated.csv
```

### 6. Train Public-Initialized `M_R0`

```bash
python -m src.train.pretrain_public \
  --processed outputs/processed/public_cluster_split.csv \
  --features outputs/processed/public_features.csv \
  --embeddings outputs/esm3_embeddings/public_peptides.npz \
  --embedding-metadata outputs/esm3_embeddings/public_peptides_metadata.csv \
  --pore-scores outputs/teacher_scores/pore_public.csv \
  --checkpoint-dir outputs/checkpoints/M_R0
```

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
  --output outputs/checkpoints/M_R0/target_affinity_expert.pt
```

### 7. Fine-Tune `M_R0 -> M_R1`

```bash
python -m src.train.finetune_round \
  --round-csv outputs/processed/private_R0.csv \
  --features outputs/processed/private_R0_features.csv \
  --embeddings outputs/esm3_embeddings/private_R0_peptides.npz \
  --embedding-metadata outputs/esm3_embeddings/private_R0_peptides_metadata.csv \
  --pore-scores outputs/teacher_scores/pore_private_R0.csv \
  --from-checkpoint outputs/checkpoints/M_R0/membrane_expert.pt \
  --output outputs/checkpoints/M_R1/membrane_expert.pt
```

For later rounds, replace the private round input and checkpoint paths according to the model-state transition:

```text
private/Rk -> M_R(k+1)
```

### 8. Score Candidates

Predict target-affinity evidence:

```bash
python -m src.score.predict_target_affinity \
  --checkpoint outputs/checkpoints/M_R0/target_affinity_expert.pt \
  --peptide-features outputs/processed/private_R0_features.csv \
  --peptide-embeddings outputs/esm3_embeddings/private_R0_peptides.npz \
  --peptide-embedding-metadata outputs/esm3_embeddings/private_R0_peptides_metadata.csv \
  --target-embeddings outputs/esm3_embeddings/targets.npz \
  --target-embedding-metadata outputs/esm3_embeddings/targets_metadata.csv \
  --peptide-id-column private_record_id \
  --aggregate-output outputs/teacher_scores/model_private_R0_aggregated.csv
```

Compute final gated consensus scores:

```bash
python -m src.score.score_candidates \
  --candidates outputs/processed/private_R0.csv \
  --embeddings outputs/esm3_embeddings/private_R0_peptides.npz \
  --embedding-metadata outputs/esm3_embeddings/private_R0_peptides_metadata.csv \
  --id-column private_record_id \
  --amp-prior outputs/checkpoints/M_R0/amp_prior.pt \
  --membrane outputs/checkpoints/M_R1/membrane_expert.pt \
  --target-affinity outputs/teacher_scores/model_private_R0_aggregated.csv \
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
peptide_id,S_aff,top_target_id,top_target_score,target_count,teacher_model
```

Candidate scores:

```text
rank
candidate id
sequence
S_prior
S_mem
S_aff
g_prior
E_comp
E_syn
E_mech
p_cons
top_target_id
top_target_score
```

## Reproducibility Notes

- ESM3 embeddings should be generated with the same ESM3 checkpoint and device settings for a given experiment.
- `outputs/processed/processed_metadata.json` records processed data counts and provenance.
- The in-repository public split uses deterministic length-aware identity clustering. If exact manuscript clustering is required, replace this stage with the same external clustering tool and threshold used in the original experiment.
- TPepPro may require a separate legacy environment. MDRAMP consumes converted pair-level TPepPro predictions through a stable CSV schema.
- `outputs/` contains generated artifacts and can be excluded from version control except for small release examples.

## Closed-Loop Usage Rule

Private round `Rk` can only update the next model state, `M_R(k+1)`.

It must not be used for same-round:

- candidate ranking
- threshold adjustment
- candidate replacement
- calibration
- hyperparameter selection

## Citation

If you use this repository, please cite the associated manuscript or preprint when available.

## License

Add the final open-source license before public release.
