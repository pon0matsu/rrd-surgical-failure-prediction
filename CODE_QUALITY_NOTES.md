# Code Quality Notes

## Bottom Line

This repository should be presented as analysis-used research code for
reproducibility, not as a polished production software package.

## Main Quality Caveat

The submitted notebook-derived `.py` files come from historical Colab
notebooks beginning at Step 03.  They preserve the imputation/modeling workflow
but contain notebook-era patterns such as Colab setup cells, originally
hard-coded output paths, repeated model-specific cells, and figure/SHAP code
mixed with model fitting.  Private path prefixes have been generalized for the
public release.  Outputs and generated artifacts are not included.

This is acceptable for provenance if clearly labeled.  It should not be framed
as a polished package API.

## More Defensible Layer

The `reviewer_response_modeling/` Python scripts are more structured and are
the better entry point for reviewer-revision rerun logic.  They are still
research scripts: long, step-specific, and designed to write local/private
intermediate outputs when data are available.

## Reader Guidance

- Use `reviewer_response_modeling/` for provenance-level reviewer-revision code.
- Use `submitted_primary_workflow/notebook_code_py/` for submitted
  workflow provenance.
- Use `REPRODUCIBILITY_AND_PROVENANCE.md` to interpret the public/private
  boundary and rerun expectations.
