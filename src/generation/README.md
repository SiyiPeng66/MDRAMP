# V6 generation

Run from the repository root. The public input CSV must include `sequence,split,cluster_id`. Validation clusters must be disjoint from training clusters.

```
python -m src.generation.cli pretrain --data public.csv --output generator.pt --seed 11
python -m src.generation.cli reinforce --checkpoint generator.pt --reference generator.pt --scorer-config scorer.json --steps 100 --beta-kl 0.1 --learning-rate 0.0001 --batch-size 128 --weight-decay 0.0001 --reference-sequences references.csv --output generator_rl.pt --seed 11
python -m src.generation.cli sample --checkpoint generator_rl.pt --output proposals.csv --seed 11
```

The step count and KL coefficient above are illustrative engineering choices, not archived manuscript values; these and RL optimizer parameters must be supplied explicitly. `--learning-rate`, `--weight-decay`, `--batch-size`, `--epochs`, `--patience`, and architectural options are available through `--help`. RL uses temperature 1 and full legal action support. Proposal sampling defaults to temperature 1 and top-p 0.95. Invalid trajectories remain in RL batches with zero reward. Scoring failures abort explicitly. The reference checkpoint must always remain the original supervised checkpoint across rounds, while `--checkpoint` can point to the preceding policy. FrozenSequenceScorer computes complete-sequence rewards using the frozen scoring configuration.

For local reconstruction pretraining:

```
python -m src.generation.pairs --input public.csv --output pairs.csv --seed 11 --edits 1
python -m src.generation.cli pretrain --conditional --data pairs.csv --output conditional.pt --seed 11
```

The perturbation count is explicit because V6 does not prescribe it. Conditional RL and sampling require `--parent STANDARD_AMINO_ACID_SEQUENCE`. Parent and padding tokens never contribute to the supervised objective or policy sequence probability. RL requires a nonempty `--reference-sequences` CSV. Conditional RL additionally requires `--parent-family family.csv` containing the designated parent and registered family sequences. Only exact registered family entries are exempted from the reference list; all other references remain enforced. Both files and the scorer configuration are hashed in the output checkpoint, along with complete CLI arguments and frozen scorer artifact hashes. Proposal rows preserve duplicates for auditing; downstream selection must deduplicate and enforce set-level diversity. This module does not treat synthetic pairs as MIC measurements.

## Audited generation and conditional workflow

The runnable public conditional pipeline is `configs/public/conditional_generator_workflow.yaml`: public reconstruction pairs → validation-selected conditional SFT → 200 parent-conditioned proposals. AMP-LO0 is `LRKKWLRFLLKFM` (R3, source ID `pep-52`). The workflow's sampled checkpoint is supervised, so its proposals are **not** reported as RL-optimized or experimentally improved. Run it through the repository's `main.py workflow` entry point. It preserves the completed unconditional checkpoint.

`sample` writes every proposal with `eligible`, `eligibility_reason`, `duplicate_proposal`, and `candidate_pool_member`, plus a hashed `.manifest.json`. Add `--eligible-output eligible.csv` to save only unique eligible proposals. Eligibility requires `--reference-sequences`; without it proposals are explicitly unassessed and cannot enter a final candidate pool. Set-level diversity and ranking remain separate selection stages. Conditional eligibility additionally requires a registered family and either a typed reference CSV (`split` column) or `--protected-sequences`. Independent `test`/`challenge` sequences are never exempted, even if they also occur in the family file.

RL writes an atomic last-step checkpoint (`--training-state`), per-trajectory JSONL (`--training-log`), and an explicit failure report when scoring fails. `--resume-training` restores model, optimizer, random states and committed logs; it rejects changed inputs, scorer artifacts, runtime, implementation, optimizer settings or output contents. Extending `--steps` is supported. Each scoring failure is distinguished from an ineligible zero-reward trajectory. The official scorer is initialized and checked before any update, including when all proposed trajectories would be ineligible; the validated MIC requirement remains enforced.

A reference must have verifiable validation-selected supervised-pretraining metadata. Legacy SFT checkpoints with recorded validation NLL, epoch, pretrain arguments and data hash are accepted without mutation. RL descendants record the original reference SHA-256; subsequent rounds must retain that same reference. Both policies use this implementation's identical legal-action support and dropout-disabled probability evaluation. `--threads` bounds PyTorch CPU threads (default 2).
