# Review macOS CPU Environment

This folder defines the reviewer-response rerun environment for the compact
`Review/` reanalysis scripts.  The environment was prepared for macOS Apple
Silicon CPU execution, including MacBook Pro M-series machines.

The intended execution mode is CPU.  The submitted Colab notebook used
`device='cuda'` for TabPFN, but the Review scripts default to
`REVIEW_TABPFN_DEVICE=cpu` and load archived TabPFN model objects with CPU
mapping where needed.

## CPU/CUDA Boundary

For the classical models used in the reviewer-response scripts
(logistic/ridge logistic regression, random forest, XGBoost, and LightGBM),
CPU versus CUDA is not expected to change the intended modeling specification.
The important reproducibility controls are package versions, fixed random
states, the same train/validation folds, the same feature set, and the same
input data.

For TabPFN, the submitted workflow used CUDA in Colab, while the
reviewer-response reruns default to CPU for portability.  Small floating-point
differences can occur across CPU/GPU backends and package builds, especially
for neural-network inference.  The public release therefore records both:

- the submitted model-construction environment and adopted hyperparameters;
- the reviewer-response macOS CPU rerun environment and `REVIEW_TABPFN_DEVICE`
  setting.

This makes the provenance explicit without requiring readers to have a CUDA
machine for reviewer-response code inspection or CPU reruns.

## Create Environment

```bash
cd <repository-root>
conda env create -f reviewer_response_modeling/00_ENVIRONMENT/environment-macos-cpu.yml
```

## Activate And Check

```bash
conda activate rrd_review_macos_cpu
export REVIEW_TABPFN_DEVICE=cpu
export MPLCONFIGDIR=/tmp/matplotlib
python reviewer_response_modeling/00_ENVIRONMENT/check_review_environment.py
```

## Optional TabPFN Checkpoint

Steps that only read submitted predictions do not need a TabPFN checkpoint.
Steps that refit TabPFN models, such as Steps 42, 47, 48, and 49, need the
TabPFN v1 checkpoint.  Prepare it once with:

```bash
python reviewer_response_modeling/00_METHOD_REFERENCE/prepare_tabpfn_checkpoint.py
```

This writes the checkpoint under
`reviewer_response_modeling/00_METHOD_REFERENCE/tabpfn_model_base/` unless
`REVIEW_TABPFN_MODEL_BASE_PATH` is set.

## Files

- `environment-macos-cpu.yml`: human-maintained environment specification.
- `check_review_environment.py`: import, CPU-device, and TabPFN checkpoint check.
- `README.md`: this reviewer-response environment note.

The internal workspace also kept a local resolved conda export.  It is not
included in the public code release because it is platform/solver specific and
contains local environment-prefix information; the human-maintained YAML plus
the environment check are the intended public reproducibility files.

## Run Review Package

```bash
cd <repository-root>
export REVIEW_TABPFN_DEVICE=cpu
export MPLCONFIGDIR=/tmp/matplotlib
python reviewer_response_modeling/<step_script>.py
```

The internal `Review/` folder is not a public release as-is.  It can include
private intermediate or patient-derived files.  The public code release keeps
only source code, environment notes, adopted hyperparameter records, and
non-row-level metadata.
