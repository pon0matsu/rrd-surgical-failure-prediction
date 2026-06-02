# Reproducibility And Provenance

## Source Of Record

Reported numeric results should be traced to the analysis-used source:

- `submitted_primary_workflow/notebook_code_py/`
- `reviewer_response_modeling/`
- `PUBLIC_SOURCE_MANIFEST.csv`

The staged `.py` files in `reviewer_response_modeling/` are copied
byte-for-byte from the analysis workspace.  Submitted imputation/modeling
notebooks from Step 03 onward are represented as `.py` code-cell exports;
outputs and execution metadata are not included, and the public exports have
minimal readability/privacy sanitization (English explanatory comments and
generalized private path prefixes).  The manifest preserves the original
notebook-code hash alongside the sanitized release-file hash.  Upstream
submitted Steps 01-02 are represented by a markdown exclusion note because they
require private patient-level source/cohort-preparation data.

## How To Interpret This Repository

This is a code-only reproducibility repository.  It contains the source code and
non-row-level metadata needed to understand and rerun the model-building
workflow when the private input data are available.  It does not include
patient-level source data, derived row-level data, prediction files, serialized
models, TabPFN checkpoints, or generated figures/tables.

## Recommended Reproducibility Check

To reproduce the analysis with authorized data access:

1. Recreate the relevant environment from `ENVIRONMENT_AND_HYPERPARAMETERS.md`.
2. Use the private input data with the same schema, outcome definition, folds,
   RFECV36 feature list, random states, and TabPFN checkpoint.
3. Provide the authorized private pre-imputation analysis CSV expected by Step
   03, then run the submitted notebook-derived `.py` exports after adapting
   `analysis_outputs` to the local authorized data/output directory, or run the
   relevant `reviewer_response_modeling/` scripts.
4. Compare regenerated aggregate metrics with the manuscript or reviewer
   response tables.
5. Record package versions, command history, and the source release tag.

For deterministic classical models, exact or near-exact equality is expected
after matching inputs and software versions.  TabPFN and GPU-backed libraries
may require small numerical tolerances depending on hardware and package
versions.

## Suggested Public Wording

> The repository contains analysis-used source code and non-row-level
> reproducibility metadata for the RRD prediction model.  Patient-level data and
> generated artifacts are not included.
