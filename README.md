# Organ-system early warning after acute myocardial infarction: analysis code

Code and protocol history for:

> An organ-system-constrained model for interpretable early warning after acute myocardial infarction. *Manuscript under review.*

**No patient data are included in this repository.** MIMIC-IV v3.1, MIMIC-IV-ECG v1.0 and eICU-CRD v2.0 are available to credentialed users through PhysioNet under their data-use agreements. Users must obtain access themselves.

## Contents

| Folder | What it does | Paper section |
|---|---|---|
| `protocol/` | Analysis protocol with amendments 1–3 and the correction note (`FULL01_PROTOCOL.md`), freeze note written before the held-out period was opened (`FREEZE_NOTE.md`), amendments 3–7, and hashes of model files and predictor matrices recorded before any model was applied to eICU-CRD (`PRESCORE_HASHES.txt`) | Methods (Protocol); Supplementary Information |
| `code/01_extraction/` | Cohort and table extraction from local MIMIC-IV files (`run_local.bat` runs it on Windows) | Cohort and study periods |
| `code/02_endpoint_features/` | Episode endpoint (v2), planned-support rules, window sensitivity, predictor construction | Prediction times and outcome; Predictors |
| `code/03_models_validation/` | Model fitting (organ-system interaction constraints, bagging, recalibration), development cross-validation, held-out validation and its analyses (including the organ-system decomposition), Table 1 | Models; Development, freezing and validation; Results |
| `code/04_additional_analyses/` | Additional analyses specified in amendments 3, 6 and 7 (amendment 7 Part B also uses the eICU-CRD files from 05) | Results; Supplementary Information |
| `code/05_external_eicu/` | eICU-CRD cohort and predictor mapping, final model refit on all MIMIC-IV data, prespecified external validation and post hoc addenda | External validation; Supplementary Information |
| `code/06_figures/` | Figures 1–6 and graphical abstract | Figures |
| `code/pilot/` | Pilot-cohort analyses (endpoint development, comparator models, internal review analyses). The pilot predictors contained a slope-calculation error that was corrected in the main analysis | Supplementary Information (pilot cohort) |

Order of execution: 01 → 02 → 03 → 04; 05 requires the outputs of 02–03; 06 last.

## Environment

Python 3.10 or later with numpy, pandas, scipy, scikit-learn (≥1.2, for `interaction_cst` in `HistGradientBoostingClassifier`), matplotlib; `interpret` (EBM comparator) and `torch` (GRU comparator) for the pilot comparators only.

File paths are hard-coded to the directories used during the analysis (for example `/mnt/user-data/uploads/...`). Edit the path constants at the top of each script to point to your local PhysioNet files.

## Licence

MIT (see `LICENSE`).
