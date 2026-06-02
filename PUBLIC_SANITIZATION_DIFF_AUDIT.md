# Public Sanitization Diff Audit

This note records the local audit used to check whether the public
`submitted_primary_workflow/notebook_code_py/` exports changed the executable
analysis implementation relative to the historical submitted notebooks in the
source archive `20250725_RRD-20260507T074230Z-3-001/20250725_RRD/`.

## Scope

- Compared historical submitted notebooks Step 03 through Step 10 with the
  public `.py` code-cell exports.
- Steps 01-02 were not compared as public code because they are upstream
  source/cohort preparation and are intentionally excluded from this code-only
  release.
- Patient-level CSVs, prediction exports, model binaries, figures, and other
  generated outputs were not inspected or summarized.

## Normalization Used For The Implementation Check

The implementation comparison removed changes that do not alter executable
analysis logic:

- code-cell separator lines in the `.py` exports
- blank lines
- full-line and inline comments
- private historical output path prefixes, mapped to `analysis_outputs`

After this normalization, the remaining executable lines were compared in
order.

## Result

| Public `.py` export | Executable implementation match |
| --- | --- |
| `03_Imputation_Dataset_Development.py` | yes |
| `04_feature_selection.py` | yes |
| `05_Hyperparameter_Tuning.py` | yes |
| `06_Undersampling.py` | yes |
| `07_CV_ORIGINAL.py` | yes |
| `08_CV_UnderSampling.py` | yes |
| `09_Final_ORIGINAL.py` | yes |
| `10_Final_UnderSampling.py` | yes |

All Step 03-10 public exports match the historical notebook executable
implementation after excluding comments, cell markers, blank lines, and the
public path-prefix generalization.

## Metadata Checks

- The first column of `submitted_primary_workflow/rfecv_features.csv` exactly
  matches the source archive feature list.  A second public-readable column was
  added with English feature labels from Table S2.
- `submitted_primary_workflow/requirements.txt` was updated from a later
  Mac-oriented portability note to a public-facing summary aligned with the
  historical Google Colab submitted model-construction environment.  The full
  observed dependency record remains in
  `reviewer_response_modeling/00_METHOD_REFERENCE/submitted_model_construction_python_environment.csv`.
- `scripts/check_release_integrity.py` passed after the sanitization edits.

## Interpretation

The public release changes are limited to public-facing sanitization and
readability edits: English explanatory comments, generalized private output
path prefixes, removal of a private local wheel-path comment, and documentation
updates.  No executable modeling, imputation, feature-selection,
under-sampling, cross-validation, final-model, metric, plotting, or SHAP logic
was changed in the Step 03-10 submitted workflow exports.
