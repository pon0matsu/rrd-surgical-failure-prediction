# Environment And Hyperparameters

This repository intentionally includes environment and hyperparameter metadata.
These are not private patient-level artifacts, and they are important for
reproducibility.

## Environment Files

- `submitted_primary_workflow/requirements.txt`
  - Public-facing requirements note aligned to the historical submitted
    Google Colab model-construction environment.
  - The private local wheel-path comment from a later portability note is not
    included because it was not analysis code.
- `reviewer_response_modeling/00_METHOD_REFERENCE/submitted_model_construction_python_environment.csv`
  - Submission-time Python package versions supplied from the model-construction
    environment.
- `reviewer_response_modeling/00_ENVIRONMENT/README.md`
  - Reviewer-response environment note.
- `reviewer_response_modeling/00_ENVIRONMENT/environment-macos-cpu.yml`
  - Reviewer-revision CPU/macOS environment used for rerun-oriented scripts.
- `reviewer_response_modeling/00_ENVIRONMENT/check_review_environment.py`
  - Lightweight environment check.

## Submitted Model-Construction Environment

From the supplied submission-time package output:

- Python 3.10 environment under `/usr/local/lib/python3.10/dist-packages`
- TabPFN 0.1.11
- numpy 1.25.2
- pandas 2.0.3
- scipy 1.11.4
- scikit-learn 1.2.2
- imbalanced-learn 0.10.1; the Colab log also reported the imblearn 0.0
  meta-package
- torch 2.3.1+cu121
- SHAP 0.46.0
- matplotlib 3.7.1 and japanize-matplotlib 1.1.3
- joblib 1.4.2 and threadpoolctl 3.5.0

The full package/dependency record is stored in
`reviewer_response_modeling/00_METHOD_REFERENCE/submitted_model_construction_python_environment.csv`.

## Reviewer-Response Rerun Environment

The reviewer-response scripts are staged under `reviewer_response_modeling/`.
Their environment files are kept in
`reviewer_response_modeling/00_ENVIRONMENT/`:

- `README.md`: reviewer-response execution note.
- `environment-macos-cpu.yml`: human-maintained macOS CPU conda environment.
- `check_review_environment.py`: import/device/checkpoint check.

## Key Reviewer-Rerun Package Versions

From `environment-macos-cpu.yml`:

- Python 3.10
- numpy 1.25.2
- pandas 2.0.3
- scipy 1.11.4
- scikit-learn 1.2.2
- imbalanced-learn 0.10.1
- xgboost 2.1.0
- lightgbm 4.1.0
- pytorch 2.3.1
- tabpfn 0.1.11
- miceforest 5.7.0
- matplotlib 3.7.1
- joblib >=1.4,<2

The submitted workflow requirements file is a public-facing summary aligned to
the submission-time Colab package output above.  The full package/dependency
record is retained in the CSV listed above.  The reviewer-revision scripts use
the documented reviewer environment above.

## Hyperparameter Metadata

Detailed submitted model settings are in:

- `reviewer_response_modeling/00_METHOD_REFERENCE/submitted_model_hyperparameters.csv`
- `reviewer_response_modeling/00_METHOD_REFERENCE/submitted_model_construction_python_environment.csv`
- `reviewer_response_modeling/review_modeling.py`

Important settings:

- Step05 grid search: `GridSearchCV(cv=5, scoring='roc_auc')`
- Step05 tuning split: `train_test_split(test_size=0.30, random_state=42)`
- Random under-sampling: `RandomUnderSampler(random_state=42)`
- TabPFN: `N_ensemble_configurations=32`
- Review rerun TabPFN default: CPU device, seed 0, with a user-supplied TabPFN
  checkpoint path if needed
- Logistic regression: `C=0.001`, `penalty='none'`, `solver='saga'`
- Random forest original condition: `max_depth=5`, `n_estimators=200`
- Random forest random-under-sampling condition: `max_depth=10`,
  `n_estimators=100`
- XGBoost original condition: `learning_rate=0.01`, `max_depth=3`,
  `n_estimators=200`
- XGBoost random-under-sampling fitting condition: `learning_rate=0.01`,
  `max_depth=7`, `n_estimators=50`
- LightGBM original condition: `learning_rate=0.01`, `max_depth=5`,
  `n_estimators=50`
- LightGBM random-under-sampling condition: `learning_rate=0.01`,
  `max_depth=5`, `n_estimators=100`
- Ridge logistic regression sensitivity: `Cs=np.logspace(-4, 4, 9)` with
  inner stratified CV

Raw grid-search result exports and score tables are intentionally not included.
The hyperparameter CSV records the Step05 search configuration and the
hyperparameters actually used in the Step07-10 model-fitting notebooks.  For
random-under-sampling XGBoost, the adopted fitting value follows Step08/Step10:
`learning_rate=0.01`, `max_depth=7`, and `n_estimators=50`.

## What Is Not Included

- Patient-level data
- Imputed matrices and fold CSVs
- Prediction files
- Serialized model binaries
- TabPFN checkpoint files
