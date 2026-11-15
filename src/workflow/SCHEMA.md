# Workflow schema version 1

Use `python -m src.workflow.run --config workflow.yaml --run-dir runs/example`.
`--plan` (alias `--dry-run`) validates and prints the resolved commands without
creating outputs. `--resume` is required to reuse an existing run.

```yaml
schema_version: 1
workspace: ..  # relative to the configuration file directory
steps:
  - name: features
    module: src.features.physicochemical
    args:
      - --input
      - {path: data/sequences.csv}
      - --output
      - {path: outputs/features.csv}
    inputs: [data/sequences.csv]
    outputs: [outputs/features.csv]
```

All stages must explicitly declare input and output artifacts. A path argument
object resolves against `workspace` and must name a declared input or output; ordinary argv strings retain their literal
value and execute with `workspace` as cwd. Absolute paths are supported. Inputs
from earlier stage outputs are permitted; forward references are rejected.
Outputs must not overlap inputs, other outputs or the run directory. Directory
artifacts are hashed recursively. Symbolic links within artifacts are rejected.
Stages must use a module in `ALLOWED_MODULES`; arbitrary shell commands and legacy
expert fine-tuning are excluded.

Each attempt writes an atomic workflow manifest and a separate finalized log.
Missing declared output is a failure even if the process exits successfully.
Concurrent use of the same run directory is rejected using an OS advisory lock.
Resume verifies normalized configuration, source code, Python identity and
completed input/output hashes. Failed stages may retry only if their recorded
inputs and partial outputs are unchanged. Outputs from an unaudited hard crash
are rejected rather than silently reused. Changing failed-stage input data
requires a new run directory and new output destinations. Existing outputs are
never overwritten by a fresh run. Explicit resume may overwrite the recorded
partial outputs of the retried failed stage.

Declare every artifact read/written by the stage, including checkpoints,
reference policies, scalers and configuration files. The runner cannot infer
undeclared accesses inside arbitrary allowed Python implementations. Logs and
manifests are control metadata, separate from stage outputs. Full model and
external environment reproducibility still depends on pinned dependency and
artifact manifests maintained by the individual stages.
