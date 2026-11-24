# Amendment 6: subgroup performance by sex, age and race or ethnicity (7 Nov 2026)

Written and saved before any subgroup result was computed. Post hoc with respect to the held-out and external results, which had already been seen; reported as post hoc. Nothing here changes any model, endpoint, predictor, threshold or previously reported result.

**Rationale.** TRIPOD+AI items 3c, 14 and 25 ask how performance varies across sociodemographic groups. The manuscript currently lists this as not assessed.

**Data.** Held-out MIMIC-IV period (2017–2022), scores already saved by the frozen held-out run (heldout_scores.npz; GRACE and NEWS2 from full_common.py). eICU-CRD, scores of the final MIMIC-IV models applied without retraining (eicu_scores.npz). No refitting.

**Subgroups.**
- Sex: female, male (MIMIC-IV predictor `female`; eICU-CRD patient.gender).
- Age at admission: <65, 65–79, ≥80 years (MIMIC-IV predictor `age`; eICU-CRD patient.age, "> 89" taken as 90).
- Race or ethnicity: MIMIC-IV admissions.race grouped as White (contains "WHITE"), Black (contains "BLACK"), other or unknown (all else); eICU-CRD patient.ethnicity grouped as Caucasian, African American, other or unknown.

**Models and measures.** Organ-system, unconstrained and logistic regression models, GRACE and NEWS2 (eICU-CRD: same set, final models). For each subgroup: assessments, positive assessments, patients with events, AUROC with a patient-level bootstrap 95% CI (1,000 resamples, seed 2026), AP, and for the organ-system model mean predicted risk against observed rate. Between-group differences in organ-system AUROC (female − male; Black − White in MIMIC-IV, African American − Caucasian in eICU-CRD; ≥80 − <65) with paired patient-level bootstrap CIs.

**Interpretation rule (fixed now).** Descriptive only; no hypothesis tests or claims of equivalence. Subgroups with fewer than 20 patients with events are flagged as imprecise. Results are reported whatever they show; no model will be changed in response.
