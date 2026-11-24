# Amendment 7: explanation compactness and local updating with few patients (10 Nov 2026)

Written and saved before any analysis below was run. Post hoc with respect to all held-out and external results, which had been seen; reported as post hoc. No existing model, endpoint, predictor, threshold or reported result is changed.

**State before writing.** The cloud working files had been lost, so the MIMIC-IV landmarks and predictor matrix were rebuilt from the extracted tables with the unchanged scripts (endpoint_full.py, features_v2fix.py). The rebuild reproduced the development counts (62,054 assessments, 615 positive, 204 patients with events) and, with the stored development models, the reported held-out results exactly (organ-system AUROC 0.873, AP 0.191; unconstrained 0.870, 0.139). The eICU-CRD predictor matrix is being rebuilt with the unchanged build_eicu.py and will be checked against the stored scores before use. The time of one model fit was not measured before this text was written.

## Part A. Explanation compactness (held-out MIMIC-IV, 2017–2022)

**Question.** How many terms does a reader need to account for a prediction under each model?

**Models.** Development-period models as evaluated in the held-out analysis: organ-system (stored), unconstrained (stored) and additive (refitted on the development period with the frozen procedure and its selected hyperparameters; the refit must reproduce the reported held-out AUROC 0.867 and AP 0.182 to three decimals, otherwise this is reported and the additive model is omitted).

**Terms.** Organ-system model: the 11 system and 8 pair terms of the exact decomposition. Additive model: one term per measurement (all summaries of the same measurement), 83 terms. Unconstrained model: every leaf value is credited to the set of organ systems used on its path, giving one term per distinct system set (exact; this is the pilot analysis repeated in the main cohort). All terms are bag-averaged and recalibrated as in the paper and centred on their mean over development assessments.

**Measures, per held-out assessment.** K80 and K90: the smallest number of terms whose absolute centred contributions add up to at least 80% and 90% of the total absolute centred contribution. Reported as median (interquartile range) over all 44,121 held-out assessments and over assessments above each model's frozen alert threshold. For the unconstrained model also: the number of distinct system-set terms, and the share of the summed variance of its terms that comes from sets of three or more systems. Descriptive; no tests.

## Part B. Local updating with few local patients (eICU-CRD)

**Question.** The paper shows that the MIMIC-IV model does not transfer unchanged (AUROC 0.615) and that the structure refitted on half of the eICU-CRD hospitals does (0.752). How many local patients are needed, and does keeping the MIMIC-IV data in the refit ("pooled updating") reduce that number?

**Design.** Same random halves of hospitals as addendum 5d (seed 2026). For each direction (fit in one half, test in the other) and each of 3 repeats, the training-half patients are randomly ordered (seed 9000 + 10·repeat + direction) and the first n = 250, 500, 1,000 and 2,000 patients are taken (nested samples). Of these, 15% of patients (split_val rule, seed as in the paper) form the local validation set used for early stopping and logistic recalibration; this is the only use of validation data.

**Update strategies**, each for the organ-system and the unconstrained model, with the hyperparameters frozen in MIMIC-IV and 5 bagged models:
- *Local only:* fit on the remaining 85% of the n local patients (as addendum 5d; predictors with no local values set constant).
- *Pooled:* fit on all 106,175 MIMIC-IV assessments plus the same local training patients, with local assessments weighted so that their total weight equals that of the MIMIC-IV assessments.
Reference rows: MIMIC-IV final model unchanged, and the full-half refit of addendum 5d.

**Measures.** Out-of-fold predictions over both halves, per repeat: AUROC (primary) and AP; mean over the 3 repeats. Mean predicted against observed risk for the pooled organ-system model.

**Comparisons**, each as the difference in repeat-averaged AUROC with a 95% CI from 1,000 patient-level bootstrap resamples (seed 2026):
1. Primary: pooled − local only, organ-system model, n = 500.
2. The same at n = 250, 1,000 and 2,000.
3. Pooled organ-system − pooled unconstrained at each n (cost of the constraint under updating).
4. Pooled organ-system at each n − MIMIC-IV model unchanged.

**Interpretation rule (fixed now).** Pooled updating is described as helpful at a sample size only if the CI for comparison 1 or 2 at that size lies above zero. All sizes and both directions of result are reported; no further tuning, no additional update strategies and no change of weights will be tried in response to the results.

## Not done
An explainable boosting machine in the main cohort was considered but the software package could not be installed in the analysis environment; the additive model remains the comparator of that class.
