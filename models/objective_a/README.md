# Objective A

The final Objective A estimator is the predeclared `exposure_shift` model. For
each broad visual area it learns a six-output mean before-to-after change from
the nine unsupervised mice in a constraint-safe representation, then adds that
frozen change to a supervised mouse’s own before-session population state.

Artifacts:

- `fitted_exposure_shift_models.joblib` — four area-specific fits;
- `configuration.json` — analysis configuration and source hashes;
- `model_summary.csv` — persistence and final exposure-shift results;
- `area_skill_by_mouse.csv` and `area_contrasts.csv` — mouse-level transfer
  comparison;
- `skill_by_mouse.png` — compact final visualization.

The serialized artifact requires the local `modeling` package to deserialize.
