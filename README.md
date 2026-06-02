# RRD Surgical Failure Prediction

This repository contains public code for the model-building analyses used in
the manuscript on surgical failure prediction after rhegmatogenous retinal
detachment surgery.

## Contents

- `submitted_primary_workflow/`: Python files for the submitted imputation and
  model-building workflow.
- `reviewer_response_modeling/`: reviewer-response model rebuilding and
  sensitivity-analysis scripts.
- `submitted_primary_workflow/rfecv_features.csv`: the RFECV36 feature list
  (36 predictors selected by recursive feature elimination with
  cross-validation), with English feature labels.
- `submitted_primary_workflow/model_hyperparameters_used.csv`: hyperparameters
  used for submitted model fitting.
- `reviewer_response_modeling/*/model_hyperparameters_used.csv`:
  hyperparameters used in reviewer-response model fitting.
- `submitted_primary_workflow/requirements.txt`: package versions.

## Software

Python package versions for the model-building workflow are listed in
`submitted_primary_workflow/requirements.txt`.

Descriptive statistics, data handling, visualization, table generation, and
statistical analyses were performed using R software version 4.3.2 with the
readxl, gtsummary, flextable, tidyverse, ggsci, ggside, and exactRankTests
packages.
