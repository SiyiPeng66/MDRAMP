# Demo inputs

Three peptides from the experiments: AMP-DN01, AMP-DN05, AMP-LO18.

- `predict.fasta`, `predict.csv`, `predict.tsv`: equivalent prediction inputs.
- `experiments.csv`: 18 original-scale observations across six strains, in μg/mL; not divided by 10.
- `expected_predictions.csv`: predictions from the supplied FinalModel. Compare with absolute tolerance 0.0001 μg/mL on the demo; small numerical differences across supported devices are expected.

Two CRKP entries (DN01 and DN05, 256 μg/mL) are the user-assigned lowest positive screening doses, not quantitative MIC measurements. DN05 MRSA=32 μg/mL is the user-adjudicated value. All other entries are reported quantitative MIC values. `provenance.json` retains per-observation attribution and source/model hashes. Across-strain minima for DN01/DN05/LO18 are respectively 4/2/2 μg/mL before calibration scaling.

These peptides were used to calibrate FinalModel. This demo checks execution, not independent biological performance. It does not reproduce ranks within the separate 100-candidate sets. Running calibration changes weights; expected predictions apply to the supplied, unchanged FinalModel.
