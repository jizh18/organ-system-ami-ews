# Amendment 5: external validation in eICU-CRD (12 Oct 2026)

Written and frozen before any model was applied to eICU data. Nothing here changes the MIMIC-IV development, endpoint definition, predictors, hyperparameters or results already reported.

**What had been seen before writing.** Table structures; names and counts of laboratory tests, infusion drugs, treatment strings, respiratory-chart labels, urine-output labels and nurse-chart labels in the candidate cohort; the share of ICU stays with infusion, respiratory-chart, treatment, intake–output and laboratory records in each hospital; and the size of the candidate cohort. While counting the candidate cohort, the crude hospital discharge status was printed once (474 deaths among 8,880 candidate unit stays). No landmark-level outcome, predictor matrix or model score had been computed.

## E0. Data

eICU Collaborative Research Database v2.0 (208 US hospitals, 2014–2015). All 31 files matched the published SHA-256 checksums. eICU contains ICU data only, with times in minutes from ICU admission, and no ECG waveforms or ECG reports.

## E1. Cohort

1. **AMI unit stays:** APACHE admission diagnosis "Infarction, acute myocardial (MI)" or "MI admitted > 24 hrs after onset of ischemia", **or** a diagnosis entry containing "acute myocardial infarction" with priority "Primary". This is the eICU analogue of a principal AMI diagnosis.
2. **Patient level:** first hospital stay per patient (`uniquepid`), defined as the earliest discharge year, with ties broken by the lowest `patienthealthsystemstayid`. All ICU stays within that hospital stay are used.
3. **Adults:** age ≥ 18. Age "> 89" is coded as 90.
4. **Hospital documentation:** hospitals are included only if at least 10% of all their ICU stays (all diagnoses, not only AMI) have records in each of `infusionDrug`, `respiratoryCharting` and `treatment`. These are the sources of the endpoint. Applying this rule leaves 123 hospitals.

## E2. Time axis and landmarks

Hospital time is ICU offset minus `hospitaladmitoffset`. Landmarks fall at 6, 12, …, 336 hours after hospital admission, as in MIMIC-IV.

A landmark is kept only if all of the following hold:
- the patient is in an ICU stay at that moment;
- the landmark is at least 1 hour after ICU admission;
- the patient is alive;
- no endpoint event has occurred before it;
- no support (planned or unplanned) is running;
- it is not inside a post-cardiac-surgery episode.

eICU cannot observe the ward, so this restriction to ICU time is unavoidable. The comparison in MIMIC-IV is therefore its ICU subgroup (AUROC 0.904).

## E3. Endpoint (same components, eICU sources)

The endpoint is death, or the unplanned start of vasopressors, inotropes, mechanical circulatory support (MCS) or invasive ventilation, within 24 hours of the landmark.

- **Death:** hospital discharge status "Expired", timed at hospital discharge.
- **Vasopressors and inotropes:** `infusionDrug` records with a positive rate for norepinephrine, epinephrine, dopamine, phenylephrine, vasopressin (vasopressors), or dobutamine and milrinone (inotropes). This is the same drug set as MIMIC-IV; isoproterenol is excluded, as in MIMIC-IV.
  - An episode is a run of records with gaps of 4 hours or less. It starts at the first record and ends 60 minutes after the last.
- **MCS:** `treatment` strings for intra-aortic balloon pump. No eICU string identifies Impella or ECMO.
  - Episodes are runs of records with gaps of 24 hours or less. They end 24 hours after the last record, unless a "removal" record comes first.
- **Invasive ventilation (IMV):**
  - Sources:
    - `respiratoryCharting` ventilator-setting labels: Vent Rate, Tidal Volume (set), TV/kg IBW, Set Vt (Servo,LTV), Adult Con Setting Set Vt, Adult Con Setting Set RR, Pressure Control, Exhaled TV (machine);
    - `treatment` "mechanical ventilation" (non-invasive excluded);
    - `respiratoryCare` vent start and end offsets, when the airway is an endotracheal tube or tracheostomy.
  - Episodes are runs of these records with gaps of 8 hours or less, ending 60 minutes after the last record.
- **Planned support, mirroring MIMIC-IV v2:**
  - **Post-cardiac-surgery episode:** the whole ICU stay plus 12 hours, when the APACHE admission diagnosis contains "CABG" or a valve operation, or a `treatment` string "cardiac surgery|CABG…" is recorded at or before the first 6 hours of that stay. Landmarks inside the episode are dropped, and supports starting in it are planned.
  - **Peri-PCI:** a PCI must be documented. That means `apachePatientResult.ptcawithin24h` = 1, or a `treatment` string "cardiac angiography|with stent placement / with angioplasty" recorded within 24 hours of hospital admission. Support counts as peri-PCI planned if, in addition, it starts within 6 hours of arrival in an ICU stay that began within 30 hours of hospital admission.
  - **Present on arrival:** support first documented within 60 minutes of ICU admission is treated as started before arrival. It counts as ongoing, not as an event. This eICU-specific rule handles care started in the emergency department or catheter laboratory.

## E4. Predictors

The same 364 predictors and the same derived statistics are used: last value, change from the previous value and from the admission value, and the 24-hour minimum, maximum and slope. The code is a line-by-line port of the MIMIC-IV feature script.

Availability time is the later of the charted time and the entry or revision time.

- **Laboratory tests:** the eICU equivalents of the MIMIC-IV item map. Units were checked and match.
  - Ionized calcium values above 2.5 are converted from mg/dL to mmol/L.
  - Only "glucose" is used, not "bedside glucose", as in MIMIC-IV.
  - **Troponin I is not mapped:** MIMIC-IV used troponin T only, and the two are not interchangeable.
  - Absolute differential counts and HbA1c are not available, so they are left missing.
- **Vital signs:** `vitalPeriodic`, reduced to the last value per variable per clock hour to match MIMIC-IV charting density, together with `vitalAperiodic` non-invasive blood pressure and `nurseCharting` vital signs.
- **Glasgow Coma Scale (GCS):** eye + verbal + motor components within 8 hours. If no components are charted, "GCS Total" is used instead.
- **Oxygen:**
  - FiO2 comes from respiratory-chart FiO2 labels (converted to a fraction) and laboratory FiO2.
  - O2 flow comes from respiratory-chart "LPM O2" or "Oxygen Flow Rate". Nurse-chart "O2 L/%" values of 15 or less are read as L/min, and values of 21–100 as %.
- **Urine:**
  - Source: intake–output "Output (ml)" rows whose label contains urine, foley, urethral, void or catheter, excluding "Count" and "Occurrence".
  - Weight: admission weight.
- **ECG predictors:** missing for every eICU assessment, because eICU has no ECG data. The electrical node therefore contributes only its learned missing-value value.
- **Baseline:**
  - Age and sex as recorded.
  - Prior heart failure, chronic kidney disease (renal insufficiency or failure), diabetes and myocardial infarction are taken from `pastHistory`. In MIMIC-IV they came from earlier admissions, so this is a semantic difference that is reported.
  - Hours since admission are counted from hospital admission.

## E5. Models (no retraining on eICU)

- **Primary: the final model.** The organ-system graph model is refitted on all MIMIC-IV patients (development + later), with the hyperparameters frozen in development and no re-tuning. It uses 5 bags with Platt scaling, as before.
- **Secondary:**
  - the development-only model reported in the paper, applied unchanged;
  - unconstrained gradient boosting (FULL) and penalized logistic regression (LR), refitted in the same way;
  - GRACE and NEWS2, computed with the same code.
    - For GRACE, "ST deviation" comes from the eICU diagnosis "with ST elevation", and "elevated enzymes" means troponin T > 0.01 or troponin I > 0.04 ng/mL. These definitions favour GRACE, not our model.
- **Alert threshold:** each model's frozen development threshold, i.e. the top 10% of its internal validation split.

## E6. Analyses

**Primary.** AUROC of the final organ-system model in eICU, with a patient-cluster bootstrap 95% CI (1,000 resamples).

**Prespecified comparisons.** No multiplicity adjustment is applied, consistent with the main paper.
- E-H1: organ-system vs FULL;
- E-H2: organ-system vs LR;
- E-H3: organ-system vs GRACE;
- E-H4: organ-system vs NEWS2.

For each, ΔAUROC and ΔAP are reported with patient-bootstrap 95% CIs.

**Secondary.**
- Average precision.
- Calibration as is: mean predicted vs observed, calibration-in-the-large, calibration slope.
- Recalibration by logistic intercept and slope, cross-fitted across two random halves of hospitals, so that it is evaluated on hospitals not used to fit it.
- Alerting at the frozen threshold: share of assessments flagged, share of episodes detected and median lead time. The NEWS2 alert rate needed to detect the same share is also reported.
- Hospital heterogeneity: per-hospital AUROC for hospitals with at least 10 event episodes, summarised by a random-effects pooled estimate and 95% prediction interval. A hospital-cluster bootstrap CI is given for the primary AUROC.
- Subgroups and alternative outcomes: death-only outcome; no PCI and no cardiac surgery; cardiac ICU vs other ICU types.
- Sensitivity analyses:
  - without the 60-minute present-on-arrival rule;
  - the development-only model.

**Explanations.**
- Each eICU prediction is decomposed exactly into node and edge terms.
- Stability: Spearman correlation between term strengths (SD of each term) in eICU and in the MIMIC-IV held-out period.
- Whether perfusion–renal is again the leading term.

**Knowledge-graph replication (K2 in eICU, exploratory).**
- Procedure: repeat the Amendment-4 open-graph analysis within eICU. Hospitals are split into two random halves. The open-graph model is fitted on one half (5 bags, same hyperparameters and procedure), pair strengths are computed on the other half, then the halves are swapped and ranks averaged.
- Reported:
  - ranks of the 8 knowledge edges and the K1 permutation p-value;
  - whether perfusion–renal and perfusion–respiratory remain in the top 8;
  - whether the candidate metabolic–perfusion pair is in the top 8.
- The candidate electrical–perfusion pair cannot be tested, because eICU has no ECG data. This is stated in advance.

**Reporting.** All results are reported whatever their direction, including any loss of discrimination or calibration on transport.

---

## Addendum 5a: post hoc diagnostics (written 12 Oct 2026, after the primary eICU results were seen)

The primary results above are reported as prespecified and are not replaced. Every model, including GRACE and NEWS2, had an AUROC of about 0.6 for the composite endpoint in eICU, but about 0.8 for death alone. That pattern suggests the eICU support-start components are measured with error.

The following diagnostics were defined before being run. They are labelled post hoc wherever they are reported.

- **D1. Components.** AUROC of every model for each separate outcome:
  - circulatory support (vasopressor, inotrope or MCS);
  - invasive ventilation.
- **D2. Reference-score contrast.** The NEWS2 AUROC for the composite in MIMIC-IV ICU assessments, compared with its value in eICU. NEWS2 is fixed and not trained on either dataset, so a drop in NEWS2 points to the endpoint or the data rather than to our model.
- **D3. Corroborated endpoint.** One alternative definition, chosen to avoid circularity with the predictors (no physiological criteria are used). An event is any of the following:
  - death;
  - a vasopressor or inotrope start in `infusionDrug` with a `treatment` vasopressor or inotropic-agent entry within ±12 hours;
  - an IABP start;
  - an invasive-ventilation start supported by two different sources within ±12 hours: (respiratoryCharting ventilator settings or respiratoryCare) and (`treatment` "mechanical ventilation").

  Uncorroborated starts still count as starts for exclusion (risk set and ongoing-support rules are unchanged); they are just not counted as events. All models are scored on this outcome; no model is changed.

## Addendum 5b: further post hoc diagnostics (written 12 Oct 2026, after D1–D3 were seen)

**What D1–D3 showed.** Requiring documentation corroboration (D3) did not raise discrimination: AUROC 0.607 for the organ-system model, with GRACE and NEWS2 at about 0.60. The composite therefore remains poorly predictable even when only corroborated events are counted.

**Alternative explanation.** In MIMIC-IV, coronary procedures were identified from coded procedures on any hospital day. In eICU, a coronary procedure is recognised only within the first 24 hours. Supports started around later catheterisation or surgery would therefore count as "unplanned" in eICU, although such starts are procedure-driven rather than physiology-driven.

- **D4.** Share of first unplanned support starts that begin within 12 hours after a `treatment` entry for cardiac angiography (any type), CABG or cardiac surgery, at any time in the stay.
- **D5. Procedure-aware endpoint.**
  - Support starts within 12 hours after such an entry, at any time, are reclassified as planned.
  - Events and landmarks are rebuilt with the same code.
  - Evaluation is restricted to landmarks that also exist in the primary set, so the predictors are unchanged.
  - All models are scored. Nothing is refitted.

These are explanatory analyses. They do not replace the prespecified result.

## Addendum 5c: model revision within eICU (written 12 Oct 2026, after the K2 replication was seen)

**What the K2 replication showed.** The open-graph model fitted within eICU, on one half of hospitals and tested on the other, reached AUROC 0.746 and 0.760. The eICU composite is therefore predictable from the same predictors when the model learns from eICU. The failure lies in transporting the MIMIC-IV parameters, not in an unpredictable endpoint. This corrects the interpretation offered after D1–D3.

**D6.**
- **Arms.** Two models are refitted on eICU with the frozen MIMIC-IV hyperparameters, using the same hospital halves, the same 5-bag procedure and 2-fold cross-fitting:
  - the organ-system graph model (the same structure);
  - unconstrained gradient boosting.
- **Comparators.** These are compared on the held-out hospital halves with:
  - the open-graph model from K2;
  - NEWS2.
- **Reported.**
  - Pooled out-of-half AUROC and AP.
  - The organ-system − unconstrained difference, with a patient-bootstrap 95% CI. This tests whether "the constraint costs little accuracy" replicates in a second dataset.
  - Cross-fitted calibration.

## Addendum 5d: graph-term recalibration without retraining (written 12 Oct 2026, before running)

**Question.** Can the MIMIC-IV organ-system model be adapted to a new hospital network without retraining any tree? Only the weights of its node and edge terms are re-estimated, from a small local sample.

**Terms.** The final model's exact decomposition gives 19 terms per assessment: 11 node terms and 8 edge terms. These terms are fixed and computed from the frozen trees.

**Term-level recalibration (TR).**
- Model: logit = z_MIMIC + δ0 + Σ_k δ_k · term_k. Here z_MIMIC is the original prediction, which enters as an offset. Each term weight is therefore 1 + δ_k, so the default is the original model.
- Estimation: logistic regression with an L2 penalty on δ (C = 1, fixed in advance, no tuning; terms are used on their native logit scale, without standardisation). The penalty shrinks the weights toward the original model when the local sample is small.
- Comparators, all fitted on the same local sample:
  - (i) **as is:** no adaptation;
  - (ii) **intercept + slope recalibration** of the organ-system model. This is the only adaptation possible for a black-box score;
  - (iii) **intercept + slope recalibration of the unconstrained model (FULL);**
  - (iv) **local retraining from scratch:**
    - the organ-system structure with the frozen hyperparameters;
    - a single model with an internal 15% split for early stopping and Platt scaling, no bagging;
  - (v) **reference:** full local retraining on an entire hospital half (Addendum 5c, AUROC 0.752).

**Design.**
- The hospital halves are the same as in 5c (seed 2026).
- Local sample sizes are n = 50, 100, 200, 500 and 1,000 patients, plus the whole half. Patients are sampled at random from the training half.
- Each size is repeated 20 times (10 times for arm iv), with fixed seeds. Samples with no event are redrawn.
- For each repeat, both directions are run (fit on half A, test on half B, and the reverse). AUROC is then computed on the pooled out-of-half predictions.
- Reported for each arm and size: mean AUROC, mean AP, the 2.5th–97.5th percentile range across repeats, and calibration slope and calibration-in-the-large.
- For TR fitted on a whole half, the estimated term weights (1 + δ_k) are also reported. They show which organ systems changed between the networks.

**Interpretation.** Success is defined as TR with ≤ 500 local patients recovering at least half of the gap between "as is" (0.615) and full local retraining (0.752), with test hospitals never used for fitting. The criterion is stated in advance so that failure is reported as failure.

### 5d results (added 12 Oct 2026, after running)

**Implementation note.** The first run solved the penalised offset logistic regression by undamped Newton steps, which diverged (weights around 10², intercept around 10¹²). It was rerun with L-BFGS on the same objective; the specification is unchanged. Only the corrected run is reported.

**Result.** Mean out-of-half AUROC (2.5th–97.5th percentile across repeats). Before adaptation, the organ-system model has AUROC 0.615.

| Local patients | Term recalibration | Intercept + slope (organ-system) | Intercept + slope (unconstrained) | Local retraining |
|---|---|---|---|---|
| 50 | 0.595 (0.569–0.615) | 0.574 | 0.566 | 0.476 |
| 100 | 0.610 (0.586–0.624) | 0.596 | 0.592 | 0.505 |
| 200 | 0.614 (0.596–0.627) | 0.601 | 0.594 | 0.595 |
| 500 | 0.632 (0.617–0.648) | 0.612 | 0.610 | 0.637 (0.509–0.701) |
| 1,000 | 0.644 (0.634–0.651) | 0.613 | 0.610 | 0.701 |
| whole half | 0.654 | 0.614 | 0.612 | 0.733 |

**The prespecified success criterion was not met.** With 500 patients, term recalibration recovered 0.017 of the 0.137 gap; the criterion required half of it.

**Additional findings.**
- Term recalibration improved discrimination monotonically with sample size. Intercept + slope recalibration cannot, because it does not change the ranking.
- Term recalibration was more stable than local retraining at every sample size of 500 or fewer.

**Term weights from the whole halves** (consistent across the two halves):
- Haemodynamic–respiratory was up-weighted about 3–4 fold.
- The injury, renal and baseline terms were reversed (weights −0.3 to −1.2). These are the nodes whose inputs differ most between the networks: troponin T was measured in 84% vs 14% of assessments, and the baseline node contains hours since admission.
- Perfusion–respiratory and metabolic were down-weighted.
