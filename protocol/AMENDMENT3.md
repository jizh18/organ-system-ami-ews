# Amendment 3: supplementary analyses requested at internal review (11 Oct 2026)

Written after the held-out period was opened (10 Oct 2026, 21:45 freeze). Every analysis below is therefore **post hoc**. None of them changes the primary model, the endpoint, the hypotheses or any reported primary result. Each is labelled as a supplementary analysis in the manuscript. Interpretation rules were fixed before running them and are listed here.

All refits use the organ-system model's tuned hyperparameters: lr 0.03, 8 leaves, min_samples_leaf 40, l2 1.0. They also use the same seed (777), the same 5-bag Platt procedure, training on all development patients and scoring once on 2017–2022. Only the interaction structure (or the input, for S6) differs, so differences reflect structure and not tuning.

## S1. Knowledge vs simplicity (reviewer Q1)
- (a) **Random concept groupings.** Measurement concepts are assigned at random to 11 groups whose sizes match the real systems; 8 random group pairs are then added. This is repeated 20 times.
- (b) **Same-hyperparameter controls:**
  - within-system only;
  - unconstrained, with the same 8-leaf trees;
  - unconstrained with max_depth 2 (each path uses at most 2 splits).
- **Interpretation:**
  - If the real grouping's AP lies inside the central 90% of the random groupings, accuracy gains are attributed to simplification, not to the specific medical grouping.
  - If it lies above the 95th percentile, the grouping itself adds accuracy.
  - The structure's value for explanation is unaffected either way.

## S2. Choice of pairs (Q2)
- Leave-one-pair-out: 8 fits.
- All 55 system pairs allowed: 1 fit.
- **Interpretation:** the design is called robust if every variant is within ±0.01 AUROC of the primary model and the leading terms stay the same.

## S3. Physiological specificity of explanations (Q3)
- Positive held-out assessments are grouped by event type: respiratory (invasive ventilation), circulatory without respiratory, and death only.
- Each term's contribution is compared with matched non-event assessments (same intensive-care status and hospital day).
- **Prespecified expectation:**
  - Respiratory-containing terms (RESP, PERFUSION×RESP, HEMO×RESP) are higher before respiratory events than before circulatory-only events.
  - Haemodynamic and perfusion terms without a respiratory component (HEMO, PERFUSION, HEMO×PERFUSION, PERFUSION×RENAL) are at least as high before circulatory events.
  - Failure of this pattern counts against physiological meaning.
- **Stability check:** importance ranking (SD of contribution) is compared across the 5 bagged models (Spearman correlation).

## S4. Why the held-out period scored higher (Q4)
- Event composition by period.
- AUROC by stratum (intensive care vs ward × admissions with vs without cardiac surgery) in both periods, for all models including the untrained scores (GRACE, NEWS2).
- Held-out AUROC after weighting later-period assessments to the development distribution of these strata × hospital day (0–1, 2–3, 4+).
- **Interpretation:** if the untrained scores improve as much as the trained models and the reweighted gap shrinks, the rise is attributed to case mix and task difficulty, not to the model.

## S5. Alert burden versus existing scores (Q5)
- Alert rate that NEWS2 and GRACE would need to reach the same episode sensitivity as the organ-system model at its development threshold.
- Burden at the conventional NEWS2 thresholds of ≥5 and ≥7.
- "Near misses": of alerts not followed by an event within 24 h, the share followed by an event within 24–72 h, or in a patient who died in hospital.

## S6. Does the model learn care decisions rather than physiology? (Q6)
- (a) Discrimination for in-hospital death within 24 h only. This outcome does not depend on treatment decisions.
- (b) Masking missingness: all missing values are replaced by the development median, with no indicators, and the organ-system model is refitted. A small loss supports reliance on measured values rather than on whether a test was ordered.
- (c) Discrimination among admissions with neither cardiac surgery nor coronary intervention.
- Manual chart review of events is not possible within this project and is stated as a limitation.
