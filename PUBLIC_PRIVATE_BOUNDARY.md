# Public/Private Boundary

## Public In This Directory

- Analysis-used model-building source code.
- Submitted RFECV36 feature names with Table S2 English labels.
- Adopted hyperparameter records and public package/environment notes.
- Provenance manifest and public sanitization notes.

## Not Public

- Patient-level or row-level source data.
- Source/cohort-preparation notebooks that require private patient-level input
  data or write derived patient-level preprocessing artifacts.
- Imputed train, hold-out, and fold CSV files.
- Prediction probability CSV files.
- Serialized model binaries and TabPFN checkpoints.
- Generated figures, manuscript tables, response-to-reviewers prose, layout
  scripts, rendered documents, and logs.
- Raw grid-search result exports and score tables; adopted hyperparameters are
  recorded separately.
- Internal audit dashboards and private provenance indexes.

## Notebook Note

Submitted imputation/modeling notebooks from Step 03 onward are included as
`.py` code-cell exports: outputs, execution counts, and notebook metadata are
not included.  The public `.py` exports keep the analysis code recognizable in
GitHub, with explanatory comments translated to English and private Colab Drive
path prefixes generalized to `analysis_outputs`.  Submitted Steps 01-02 are
not included as code because they sit on the private source/cohort-preparation
side of the boundary.

## Refactor Note

Cleaned rewrite implementations are intentionally not included in this public
reproducibility release.  The goal is to show the analysis-used code rather
than a rewritten version; the only public-facing edits in the submitted
notebook exports are comment translation and private path-prefix
generalization.
