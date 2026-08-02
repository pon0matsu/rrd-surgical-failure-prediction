# Risk Stratification for Surgical Failure After Primary PPV for RRD

This repository contains public code for predicting surgical failure after primary pars plana vitrectomy (PPV) for rhegmatogenous retinal detachment (RRD).

## Scope and limitations

The model was developed for primary RRD treated with primary PPV as the retinal-detachment repair strategy. Eyes treated with scleral buckling (SB) alone or PPV combined with SB, and syndromic, traumatic, or recurrent RRD, were excluded.

The output is a risk-stratification score, not a calibrated absolute risk; the 0.7 threshold does not represent a 70% probability of failure. Discrimination was moderate, calibration was imperfect, and the external cohort comprised 200 eyes from a single US center. Further validation and local recalibration are required before clinical implementation.

The model is for research and further validation and should not be used alone for clinical
decision-making.

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
