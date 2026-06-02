# RRD Surgical Failure Prediction

This repository contains public code for reproducing the model-building
analyses used in the manuscript on surgical failure prediction after
rhegmatogenous retinal detachment surgery.

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

## Data

Patient-level source data, derived datasets, prediction files, model binaries,
figures, and manuscript output files are not included.

## Primary Model

The primary model is the submitted RFECV36 TabPFN workflow with random
under-sampling applied within each training set. Other model variants are
sensitivity or comparator analyses.

## License

MIT License.
