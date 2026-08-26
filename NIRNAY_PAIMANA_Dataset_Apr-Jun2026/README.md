# NIRNAY PAIMANA Dataset Package

## Source
Extracted from the supplied PAIMANA Flash Reports:
- April 2026
- May 2026
- June 2026

Core source table: **Table 6 — All Ongoing Projects**.

## Dataset sizes
- Monthly snapshots: 5,815 rows
- Unique project codes across all supplied months: 2,038
- April 2026: 1,981 rows
- May 2026: 1,987 rows
- June 2026: 1,847 rows

## Files
### `paimana_ongoing_monthly_snapshots.csv`
The clean source-of-truth project-month table. One row per project per supplied PAIMANA month.

### `paimana_project_master_latest_available.csv`
One row per unique project code, using the latest available supplied snapshot.

### `paimana_ml_features.csv`
Project-month data plus feature engineering aligned to the NIRNAY ML prompt:
- cost variance
- expenditure/physical progress gap
- schedule revision
- schedule slippage at snapshot
- planned duration
- progress velocity
- rolling progress velocity
- snapshot cadence
- temporal sector prior rates
- progress-stagnation warning rule
- retrospective cost/schedule revision labels

### `paimana_training_baseline.csv`
A conservative baseline-training table. Predictor columns intentionally avoid `revised_cost_cr`, `expected_completion_date`, expenditure and current physical progress so the target is not directly supplied to the classifier. `target_cost_overrun` and `target_schedule_overrun` are retrospective labels derived from the PAIMANA revision fields.

**Important:** these labels indicate whether a revision exists in the reported snapshot. They are not yet a leakage-free future-event forecasting target. For a strong SIH evaluation, use a longer monthly history and define targets using future snapshots.

### `paimana_data_dictionary.csv`
Field-level definitions and whether each field is source-derived, derived, or a rule/label.

## Important data limitations
1. No values were fabricated when the supplied PDFs did not report them.
2. Ministry and sector were propagated from the grouped headings in PAIMANA Table 6 to each project row.
3. Milestone-level records are not present in the supplied Table 6 data, so milestone delay rate was not invented.
4. Financial progress is not an official field in these extracted rows. The package derives a proxy:
   `cumulative_expenditure_cr / revised_cost_cr * 100`.
5. The supplied history is only April–June 2026. That is enough to wire the pipeline and demonstrate feature engineering, but not enough for a robust production forecasting claim.
6. The 4-quarter cost-acceleration rule from the NIRNAY prompt cannot be honestly evaluated from only three monthly snapshots.

## Recommended NIRNAY flow
PAIMANA CSV -> validation/cleaning -> feature pipeline -> baseline model -> ML model -> SHAP -> early warnings -> intervention rules -> API -> dashboard.

The current package is the data foundation. It does not contain fabricated model risk scores.
