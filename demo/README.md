# Demo

Six prediction examples use neutral identifiers:

| IDs | Class | Prediction criterion |
|---|---|---|
| Test1–Test3 | Positive | MIC ≤128 μg/mL |
| Test4–Test6 | Negative | MIC >256 μg/mL |

`predict.fasta`, `predict.csv` and `predict.tsv` contain the same six sequences. `expected_predictions.csv` contains the released FinalModel predictions; `expected_classes.csv` records the class checks. Use an absolute tolerance of 0.0001 μg/mL when comparing reference predictions.

```bash
python main.py predict --input demo/predict.fasta
```

`experiments.csv` remains a positive-only calibration example: 18 original-scale observations for Test1–Test3 across six strains. Two positive screening concentrations are assigned MIC labels; one value was adjudicated. Attribution is recorded in `provenance.json`. Do not pre-divide these values; the default calibration divisor is 10.

Test4–Test6 have MIC >256 across all six tested strains and were used as unscaled negative constraints. They are prediction examples, not included in the positive-only calibration file: applying the default divisor of 10 to their negative bounds would change the intended constraint.

All six sequences participated in model fitting. This demo verifies execution and expected outputs, not independent biological performance. Calibrating a new model changes its predictions; reference outputs apply to the released FinalModel.
