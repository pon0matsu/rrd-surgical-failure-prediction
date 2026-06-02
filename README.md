# RRD Analysis-Used Public Model Code

This directory is a public, code-only reproducibility repository for the RRD
prediction model project.  It intentionally focuses on the analysis-used source
code and non-row-level reproducibility metadata needed to trace the manuscript
and reviewer-response model-building workflow.

## What Was Preserved

- Python files for model rebuilding were copied byte-for-byte from the analysis
  workspace into `reviewer_response_modeling/`.
- Submitted imputation/modeling notebooks from Step 03 onward were converted to
  `.py` code-cell exports for readability.  Outputs, execution counts, and
  notebook metadata are not included.  For public readability and privacy,
  Japanese explanatory comments were translated and private Colab Drive path
  prefixes were generalized to `analysis_outputs`; the manifest records the
  original notebook-code hashes and the sanitized release-file hashes.
- RFECV36 feature names with Table S2 English labels, adopted hyperparameter
  records, and package/environment notes are included as non-row-level model
  metadata.  The submitted-workflow requirements note is aligned to the
  historical Google Colab execution environment, with private local portability
  comments removed.

## Code-Quality Caveat

The submitted notebooks are historical Colab notebooks from the analysis
timeline.  They are included for provenance, not presented as polished reusable
software.  They contain Colab setup commands, originally hard-coded Colab
drive paths, manual model-by-model cells, figure/SHAP code mixed with model
fitting, and commands that wrote model/prediction artifacts during the private
analysis.  Private path prefixes have been generalized in the public `.py`
exports, and the generated artifacts and patient-level data are not included
here.

For a reader looking for the reviewer-revision rerun logic, start with the
`reviewer_response_modeling/` Python scripts.  For a reader checking submitted
workflow provenance, inspect the sanitized notebook-derived `.py` exports
together with the manifest hashes.

## What Was Excluded

- Patient-level source data, cleaned data, train/hold-out CSVs, imputed fold
  CSVs, prediction CSVs, model binaries, checkpoints, caches, figures, Excel
  templates, PDFs, HTML, DOCX files, and logs.
- Standalone table/figure/response/audit scripts whose purpose is presentation
  or manuscript packaging rather than model rebuilding.
- The initial submitted source/cohort-preparation notebooks, Steps 01-02, are
  excluded because they require private patient-level source or cleaned input
  data and write derived patient-level preprocessing artifacts.

## Layout

- `submitted_primary_workflow/`: submitted imputation/modeling workflow
  notebook code exported as `.py` from Step 03 onward, plus non-row-level
  metadata and a markdown note for excluded Steps 01-02.
- `reviewer_response_modeling/`: analysis-used Review scripts for PVD-only,
  strict-endpoint, feature-set, class-imbalance, and repeated-CV model
  rebuilding.
- `reviewer_response_modeling/00_METHOD_REFERENCE/`: adopted submitted
  hyperparameters and model-construction-time Python environment records.
- `scripts/`: lightweight repository integrity helper scripts.
- `PUBLIC_SOURCE_MANIFEST.csv`: SHA-256 provenance manifest.
- `PUBLIC_SANITIZATION_DIFF_AUDIT.md`: implementation-equivalence audit for
  the public comment/path sanitization edits.
- `PUBLIC_PRIVATE_BOUNDARY.md`: public/private release boundary.
- `ENVIRONMENT_AND_HYPERPARAMETERS.md`: package/environment versions, seeds,
  and submitted model hyperparameter references.
- `REPRODUCIBILITY_AND_PROVENANCE.md`: how to interpret this code-only
  reproducibility release.
- `EXCLUDED_MATERIAL_SUMMARY.csv`: examples of intentionally excluded files.
- `LICENSE`: MIT license for the public code release.

## Primary Analysis Boundary

The manuscript primary model remains the submitted RFECV36 TabPFN workflow with
random under-sampling applied inside each training set. Logistic regression,
random forest, XGBoost, LightGBM, ridge logistic regression, all-feature,
compact-feature, no-under-sampling, class-weighting, PVD-only, strict-endpoint,
and repeated-CV analyses are sensitivity or comparator analyses.
