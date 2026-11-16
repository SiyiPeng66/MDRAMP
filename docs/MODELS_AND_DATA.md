# Models and data

## Model assets

BaseModel contains public-data-trained experts and an autoregressive generator. Its public training data have no quantitative MIC labels; the first experimental calibration initializes the MIC head. FinalModel combines a preference-trained MIC head with frozen ESM3, three experts and a 575-protein scoring panel.

`configs/assets/downloads.json` records pinned Hugging Face revisions and file checksums. Model files are verified before installation under `assets/`. Set `MDRAMP_ASSETS` to change this directory. Interrupted downloads resume when the command is rerun. Use `HTTPS_PROXY` if your network requires a proxy. A checksum error is not silently repaired: move the affected file aside and retry; a corrupt Hugging Face download cache may also need removal.

Custom models must include a companion `scorer.json` with valid asset paths. Calibration saves a separate model and updates the default registry only with `--update-final`.

## Public training

`data/public/` contains sequence tables, descriptors, recorded teacher scores, pair-curation evidence and the training plan. Training uses the recorded teacher scores; it does not regenerate them by running teacher models. Large ESM3 feature archives are excluded from the source distribution.

Missing features are computed into `data/public/.feature_cache/`. Interrupted encoding can resume. Inputs and existing archives are checksum-verified; recomputed caches are bound to sequences, model weights and encoding runtime. Rebuilt features may differ slightly numerically from historical caches, so byte-identical historical retraining is not claimed. Full encoding and training can be lengthy on CPU.

## Calibration and interpretation

Calibration aggregates observations to the best peptide-level MIC interval across strains; strain is not an inference input. Right-censored values such as `>128` retain their bounds. The default target divisor is 10, configurable with `--mic-divisor`. Incompatible duplicate observations are rejected. Table headers are case-sensitive; sequence whitespace is removed and letters are uppercased.

FinalModel was trained with adjusted labels and explicit ranking preferences. Those preferences were fitted within fixed candidate sets; ranks depend on the set being scored. Model outputs do not establish independent predictive accuracy. Demo peptides were part of calibration. The calibration demo contains original-scale observations; two screening concentrations were assigned MIC labels and one value was adjudicated.

Generation samples the pretrained autoregressive policy, applies physicochemical filters, optionally excludes sequences supplied with `--exclude` and reranks using the selected MIC model. It is not a new reinforcement-learning run per request or experimental safety validation.

## Study data

The full experimental records and derived candidate tables are provided in the manuscript supplementary materials. Only the six selected demo sequences are distributed here.

## Reproducibility

The runtime dependency versions are pinned in `configs/requirements-runtime.lock.txt`, installed through `environment.yml`. Command parameters and asset hashes are recorded with each run. Outputs never overwrite an existing results directory.

Runtime checks cover demo prediction, small calibration and generation, feature reconstruction, and unit tests on macOS ARM64. This does not constitute full historical round reproduction or independent biological validation. Linux has not been independently exercised; native Windows is unsupported (use WSL2).

Upstream model and data licenses apply separately. The Hugging Face model card describes model-asset licensing; it does not automatically grant a license to all source datasets or this source repository.

FinalModel includes negative constraints: 27 R0–R2 experimental negatives and 57 public training negatives all predict MIC >256 μg/mL. The 203 retained original sequences changed by at most 1.05%; designated ranks remain 1, 5 and 8. Only 7 of 12 public held-out sequences exceeded 256. These sequences had prior AMP-pretraining exposure and some primary-source evidence remains pending; this is an internal MIC-head holdout, not independent external validation.
