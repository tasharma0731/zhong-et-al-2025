# Modeling

## Objective A — cross-day exposure transfer

Objective A uses one six-number neuronal d-prime population summary per mouse,
area, and session. The locked estimator is `exposure_shift`: a mouse-weighted,
area-specific mean change learned in a constraint-safe representation from the
unsupervised cohort, then applied frozen to supervised before-session states.

Core implementation:

- `targets.py` — constraint-safe distribution transform;
- `estimators.py` — exposure-shift and supporting estimator definitions;
- `preparation.py` — validated modeling tables;
- `validation.py` — mouse-grouped development and frozen transfer;
- `transfer_pipeline.py` — end-to-end orchestration.

The four area-specific exposure-shift fits are in `../models/objective_a/`.

## Objective B — direct learning-boundary forecast

Objective B predicts the first and second after-learning W20 population states
directly from before-learning inputs. Selection occurs independently for each
area and horizon using unsupervised-mouse validation. The selected models are
then frozen for supervised evaluation.

Core implementation:

- `experiments/boundary_population_forecaster.py` — data contract, features,
  candidates, nested selection, fitting, and prediction;
- `experiments/run_boundary_population_forecaster_experiment.py` — runner and
  immutable output contract;
- `experiments/boundary_population_forecaster_plots.py` — final figures;
- `experiments/boundary_population_actual_accuracy.py` — original-unit error
  summaries.

Model files and evaluation tables are in `../models/objective_b/`.
