# Objective B

The final Objective B model predicts the six statistics of the first two
after-learning W20 population states from before-learning data only. Candidate
family and ridge penalty are selected separately for each area and horizon by
unsupervised-mouse validation, after which the four selected fits are frozen.

Artifacts:

- `fitted_boundary_population_models.joblib` — four final `(area, horizon)`
  fits;
- `final_selections.csv` and `configuration.json` — frozen selection contract;
- `model_summary.csv`, `mouse_comparison.csv`, and `metric_errors.csv` — compact
  evaluation tables;
- `predictions.csv` — model and persistence predictions used by the report;
- `RESULTS_REPORT.md` and `ACTUAL_VALUE_ACCURACY.md` — human-readable evidence;
- `plots/` — result figures.
