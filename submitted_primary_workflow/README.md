# Submitted Primary Workflow

This folder contains `.py` code-cell exports from the submitted imputation and
modeling notebooks, plus non-row-level metadata.

The `.py` files are exported for GitHub readability.  They preserve the
submitted notebook code-cell order, without notebook outputs, execution counts,
or notebook metadata.  For public readability and privacy, explanatory comments
were translated to English and private Colab Drive path prefixes were
generalized to `analysis_outputs`.  They are provenance files rather than
cleaned standalone scripts, so authorized users may need to adapt local data
paths before rerunning them.

The notebook numbering follows the historical submitted workflow.  Steps 01-02
were upstream source/cohort-preparation notebooks and are intentionally
excluded from this public model-code release because they require private
patient-level input data and write derived patient-level preprocessing
artifacts.  The public model-building workflow therefore begins with Step 03.
Step 03 expects the private pre-imputation analysis CSV produced by the
excluded upstream preparation steps.

`rfecv_features.csv` is intentionally headerless because the submitted code
reads it with `header=None`.  It contains 36 data rows: column 1 is the original
RFECV feature identifier used by the code, and column 2 is the English feature
label added for public readability.  The file is encoded as UTF-8 with BOM so
Japanese feature identifiers display correctly when opened directly in Excel.
