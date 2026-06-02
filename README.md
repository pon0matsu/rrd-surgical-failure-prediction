# RRD Surgical Failure Prediction

This repository contains public code for reproducing the model-building
analyses used in the manuscript on surgical failure prediction after
rhegmatogenous retinal detachment surgery. Patient-level data and generated
analysis outputs are not included.

## Contents

- `submitted_primary_workflow/`: submitted imputation and model-building
  workflow exported as Python code. Steps 01-02 are excluded because they
  require private patient-level source or preprocessing data.
- `reviewer_response_modeling/`: reviewer-response and sensitivity-analysis
  model rebuilding scripts.
- `submitted_primary_workflow/rfecv_features.csv`: RFECV36 feature list with
  English feature labels.
- `submitted_primary_workflow/requirements.txt`: package versions for the
  historical Google Colab workflow.

## License

MIT License.
