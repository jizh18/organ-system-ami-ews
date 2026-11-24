# FULL01 protocol — enlarged cohort and held-out later period (locked 9 Oct 2026, before any data extraction)

## Data
- Cohort: F05 of the raw rebuild (first admission, principal type 1/unspecified AMI, adult). Amended 9 Oct 17:55 before any outcome data were extracted: the ECG-within-6-h requirement was dropped because the previous 1,688-patient cohort was not ECG-restricted (only 1,454 met it); ECG availability is kept as a flag and a sensitivity subgroup.
- Development set: estimated real admission years entirely within 2008–2016 (includes the previous 1,688 fitting patients plus all other development partitions).
- Held-out set: years entirely within 2017–2022. Not opened until every development decision below is frozen; evaluated once.
- Same endpoint code (primary; strict and all-support as sensitivity), same features, same keep-rule fitted on development only.

## Models (unchanged code, hyperparameters re-tuned in development only)
AXIS (primary), WITHIN, FULL, penalised LR (C grid 1e-4…1e-2), GRACE/NEWS2 (secondary). Logit-averaged 5-bag ensembles, Platt recalibration on inner patients.

## Development analysis
5-fold patient-grouped CV × 3 repeats, as in EP3/REV1.

## Prespecified hypotheses (development CV and held-out, AUROC; patient bootstrap 95% CI)
H1 AXIS vs FULL: no evidence of difference (report CI).
H2 AXIS vs LR: AXIS better if CI of ΔAUROC excludes 0 (two-sided). Equally reported if LR better or no difference.
H3 AXIS vs WITHIN: cross-system pairs useful only if CI excludes 0.
H4 AXIS vs GRACE and NEWS2 in the held-out set (overall and ICU assessments).
Held-out calibration (slope, intercept), alerting with development-fixed threshold, patient-level alert burden.
No hypothesis, endpoint rule, feature or model may be changed after the held-out set is scored.

## Amendment 1 (10 Oct 2026, before the enlarged data were extracted or opened)
Reason: in the 1,666-patient analysis, sensitivity of the endpoint to window length came almost entirely from cardiac-surgery admissions (105 of 113 events added by halving the windows, 23 of 24 removed by lengthening them). A clinically anchored rule removes this.
1. Primary endpoint v2 (episode-based): for admissions with cardiac surgery (CABG/valve), the post-operative episode runs from 00:00 of the first surgery date to ICU discharge after surgery + 12 h (or surgery date + 72 h if no ICU stay). Support starts inside it are planned; landmarks inside it are excluded from the risk set. PACU (±12 h) and PCI (30 h and ICU arrival ≤6 h) rules unchanged. Sensitivity: PACU/PCI windows ×0.5 and ×1.5; non-surgical stratum. In the 1,666-patient data this gave 93 / 81 / 80 patients with events at ×0.5 / ×1 / ×1.5 (old rule: 225 / 113 / 91).
2. Added model HYBRID: equal-weight average of the calibrated log-odds of penalised LR and AXIS (no tuned weight). Exactly decomposable by system (LR terms summed per system + AXIS system/pair terms). Partners for the same construction: LR+ADD, LR+WITHIN, LR+FULL.
3. Added hypotheses (development CV and held-out, AUROC and AP with patient-bootstrap 95% CI):
   H5 HYBRID vs LR: knowledge-constrained trees add to the linear backbone if the CI of ΔAP excludes 0.
   H6 LR+AXIS vs LR+ADD: knowledge-defined relations (beyond additive non-linearity) add if the CI of ΔAP excludes 0.
   H7 LR+AXIS vs LR+FULL: reported descriptively (non-inferiority not claimed).
   These were suggested by post hoc analyses of the 1,666-patient data (HYBRID−LR ΔAP +0.026, 95% CI 0.004–0.045) and are therefore confirmatory only in the enlarged development set and the held-out 2017–2022 set.

## Amendment 2 (10 Oct 2026, before any outcome in the enlarged data was examined)
The previous 1,688-patient cohort was not restricted by the ECG ±6 h criterion (235 of 1,688 lacked such an ECG; 1,687 of 1,688 are first type-1/unspecified AMI admissions in the development period). Requiring the ECG would also discard 72% of the later period (1,933 of 2,687), where ECG linkage is sparser.
Primary population: all first qualifying admissions, development 4,224 and later period 2,687 (boundary 608 excluded); ECG features are missing when no ECG is available.
Sensitivity: the subset with an ECG within ±6 h of admission (development 3,486; later 754).
Extraction used: outputs/experiments/full_v2_20261009 (cohort_flow.csv), produced by the v2 version of extract_full.py.

## Correction to Amendment 1 (10 Oct 2026, 18:40)
Amendment 1 states it was made before the enlarged data were opened. That is not accurate for the development set: on 9 Oct a development-only cross-validation (2 of 3 repeats) under the original endpoint (v1) had already been run and examined (AXIS 0.836, FULL 0.839, ADD 0.830, LR 0.823 AUROC; 289 patients with events). The held-out 2017–2022 set had not been scored (the run started on 9 Oct did not complete and produced no output). Consequences: development-set comparisons under v2 are exploratory with respect to the v1 results; the held-out evaluation remains the confirmatory test. Both v1 and v2 development results will be reported. The slope features were corrected on 9 Oct (time origin per admission) before either development run.
