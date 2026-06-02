# Steps 01–02: Private Clinical Data Preparation

The original submitted workflow included two upstream notebooks before the
imputation and model-building steps:

- Step 01: source clinical-data cleaning
- Step 02: cohort construction and pre-imputation data preparation

These notebooks are not included in this public code release because they use
private patient-level clinical data and generate derived row-level preprocessing
artifacts. In accordance with patient privacy, institutional, and data-use
restrictions, the public repository excludes private source/cleaned data and
derived row-level preprocessing outputs, while retaining the downstream analysis
code and non-row-level metadata.

Accordingly, the `.py` code-cell export sequence starts at Step 03, preserving
the original workflow numbering and indicating that Steps 01–02 are private
clinical data-preparation steps. Running the downstream scripts on the original
clinical cohort requires non-public preprocessing outputs.