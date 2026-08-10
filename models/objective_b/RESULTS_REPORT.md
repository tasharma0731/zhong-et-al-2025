# Direct population forecast across learning

## Result in one sentence

The model uses only pre-learning information to predict the complete
six-number area-level d-prime distribution in the first and second W20
windows after learning. On the three supervised mice with the required two
pre-learning W20 windows, the frozen model reduces NRMSE versus persistence in
all four area/horizon comparisons. The result is exploratory because the
supervised test set has been inspected repeatedly.

## What is predicted

For each mouse, broad area (`mHV` or `aHV`), and future horizon, the target is:

`(q05, median, q95, SD, leaf-selective fraction, circle-selective fraction)`.

- horizon 0 is the **first after-learning W20**;
- horizon 1 is the **second after-learning W20**.

This is a population forecast. It does not predict or require registration of
the same individual neuron across days.

## Model in one picture

```text
complete before session + final two before W20s (full and SVD) + behavior
                                  |
                                  v
          constraint-safe population-state representation g(.)
                                  |
                                  v
       choose one direct residual model per area and future horizon
       using unsupervised mice only (nested leave-one-mouse-out)
                                  |
                                  v
 g(predicted after W20_h) = g(last before W20) + predicted boundary residual
                                  |
                                  v
          inverse-transform to six valid, directly comparable values
```

The candidate set contains exact persistence, a horizon-specific mean
cross-day shift, and ridge regressions using increasingly rich before-only
features. Standard ridge leaves the intercept unpenalized, so it can learn the
common exposure shift while shrinking unreliable state-dependent slopes.

## Headline performance

NRMSE is computed from errors against the **actual six values**, after scaling
each metric using training mice only. The percentage is error reduction versus
carrying the final before-learning W20 forward:

| split | area | horizon | model | mice | mean_nrmse | sem_nrmse | improvement_vs_persistence_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen supervised transfer | aHV | first after W20 | Last-before W20 persistence | 3.000 | 1.731 | 0.282 | 0.000 |
| Frozen supervised transfer | aHV | first after W20 | Direct boundary forecaster | 3.000 | 1.348 | 0.284 | 22.130 |
| Frozen supervised transfer | aHV | second after W20 | Last-before W20 persistence | 3.000 | 3.709 | 1.054 | 0.000 |
| Frozen supervised transfer | aHV | second after W20 | Direct boundary forecaster | 3.000 | 3.224 | 1.067 | 13.088 |
| Frozen supervised transfer | mHV | first after W20 | Last-before W20 persistence | 3.000 | 1.516 | 0.146 | 0.000 |
| Frozen supervised transfer | mHV | first after W20 | Direct boundary forecaster | 3.000 | 0.849 | 0.186 | 43.988 |
| Frozen supervised transfer | mHV | second after W20 | Last-before W20 persistence | 3.000 | 1.940 | 0.345 | 0.000 |
| Frozen supervised transfer | mHV | second after W20 | Direct boundary forecaster | 3.000 | 0.867 | 0.265 | 55.310 |
| Unsupervised nested LOMO | aHV | first after W20 | Last-before W20 persistence | 9.000 | 0.726 | 0.120 | 0.000 |
| Unsupervised nested LOMO | aHV | first after W20 | Direct boundary forecaster | 9.000 | 0.875 | 0.160 | -20.485 |
| Unsupervised nested LOMO | aHV | second after W20 | Last-before W20 persistence | 9.000 | 1.133 | 0.237 | 0.000 |
| Unsupervised nested LOMO | aHV | second after W20 | Direct boundary forecaster | 9.000 | 1.005 | 0.378 | 11.277 |
| Unsupervised nested LOMO | mHV | first after W20 | Last-before W20 persistence | 9.000 | 1.173 | 0.208 | 0.000 |
| Unsupervised nested LOMO | mHV | first after W20 | Direct boundary forecaster | 9.000 | 0.929 | 0.140 | 20.829 |
| Unsupervised nested LOMO | mHV | second after W20 | Last-before W20 persistence | 9.000 | 1.399 | 0.138 | 0.000 |
| Unsupervised nested LOMO | mHV | second after W20 | Direct boundary forecaster | 9.000 | 0.911 | 0.128 | 34.866 |

## Presentation figures

- [Model comparison](plots/boundary_population_model_comparison.png)
- [Predicted versus actual six-statistic values](plots/boundary_population_predicted_vs_actual.png)
- [All eligible supervised trajectories](plots/boundary_population_supervised_trajectories.png)
- [TX108](plots/boundary_population_supervised_TX108.png)
- [TX60](plots/boundary_population_supervised_TX60.png)
- [VR2](plots/boundary_population_supervised_VR2.png)
- [TX109 input-eligibility slide](plots/boundary_population_supervised_TX109.png)

## Frozen model choices

Candidate family and alpha are chosen independently for each area and horizon
from unsupervised-mouse LOMO only:

| area | horizon | selection_rule | selected_candidate | selected_feature_variant | selected_alpha | best_mean_nrmse | best_sem_nrmse | horizon_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aHV | 0.000 | minimum_cv | mean_shift | mean_shift |  | 0.699 | 0.065 | first after W20 |
| aHV | 1.000 | minimum_cv | dual_full_svd__alpha_100 | dual_full_svd | 100.000 | 0.916 | 0.372 | second after W20 |
| mHV | 0.000 | minimum_cv | mean_shift | mean_shift |  | 0.929 | 0.140 | first after W20 |
| mHV | 1.000 | minimum_cv | session_only__alpha_1000 | session_only | 1000.000 | 0.853 | 0.112 | second after W20 |

The richer full-neural + SVD representation is selected only where its
unsupervised validation error warrants it. Simpler mean-shift or
whole-session candidates remain valid outcomes rather than forcing the largest
feature set into a nine-mouse regression.

## Error against the real values

These are raw-unit errors for the frozen supervised predictions:

| area | horizon | metric | mae | rmse | mean_observed | mean_predicted | mae_improvement_vs_persistence_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aHV | first after W20 | frac_circle_selective | 0.052 | 0.056 | 0.184 | 0.236 | 47.133 |
| aHV | first after W20 | frac_leaf_selective | 0.174 | 0.184 | 0.431 | 0.257 | 21.476 |
| aHV | first after W20 | median | 0.187 | 0.203 | 0.196 | 0.008 | 26.549 |
| aHV | first after W20 | q05 | 0.040 | 0.044 | -0.736 | -0.760 | 45.137 |
| aHV | first after W20 | q95 | 0.367 | 0.380 | 1.103 | 0.736 | 15.103 |
| aHV | first after W20 | sd_dprime | 0.093 | 0.099 | 0.589 | 0.495 | 21.234 |
| aHV | second after W20 | frac_circle_selective | 0.046 | 0.050 | 0.233 | 0.253 | 15.933 |
| aHV | second after W20 | frac_leaf_selective | 0.207 | 0.209 | 0.468 | 0.260 | 19.722 |
| aHV | second after W20 | median | 0.241 | 0.246 | 0.234 | -0.007 | 17.995 |
| aHV | second after W20 | q05 | 0.229 | 0.343 | -1.010 | -0.817 | -2.857 |
| aHV | second after W20 | q95 | 0.836 | 0.903 | 1.603 | 0.767 | 10.332 |
| aHV | second after W20 | sd_dprime | 0.288 | 0.351 | 0.835 | 0.547 | 21.129 |
| mHV | first after W20 | frac_circle_selective | 0.056 | 0.060 | 0.283 | 0.339 | -9.843 |
| mHV | first after W20 | frac_leaf_selective | 0.104 | 0.108 | 0.444 | 0.340 | 38.811 |
| mHV | first after W20 | median | 0.173 | 0.198 | 0.183 | 0.010 | 22.467 |
| mHV | first after W20 | q05 | 0.097 | 0.106 | -1.374 | -1.373 | 70.100 |
| mHV | first after W20 | q95 | 0.195 | 0.199 | 1.711 | 1.516 | 73.319 |
| mHV | first after W20 | sd_dprime | 0.165 | 0.177 | 1.134 | 0.969 | 66.043 |
| mHV | second after W20 | frac_circle_selective | 0.027 | 0.033 | 0.351 | 0.325 | -28.322 |
| mHV | second after W20 | frac_leaf_selective | 0.038 | 0.041 | 0.406 | 0.367 | 70.872 |
| mHV | second after W20 | median | 0.066 | 0.075 | 0.084 | 0.073 | 46.971 |
| mHV | second after W20 | q05 | 0.282 | 0.353 | -1.753 | -1.471 | 60.024 |
| mHV | second after W20 | q95 | 0.428 | 0.509 | 2.012 | 1.584 | 58.578 |
| mHV | second after W20 | sd_dprime | 0.302 | 0.332 | 1.358 | 1.056 | 57.441 |

## Data support

| cohort | area | mice | before_sessions | minimum_before_w20_windows | maximum_before_w20_windows |
| --- | --- | --- | --- | --- | --- |
| supervised | aHV | 3.000 | 3.000 | 4.000 | 8.000 |
| supervised | mHV | 3.000 | 3.000 | 4.000 | 8.000 |
| unsupervised | aHV | 9.000 | 9.000 | 3.000 | 11.000 |
| unsupervised | mHV | 9.000 | 9.000 | 3.000 | 11.000 |

TX109 is not silently imputed: its before-learning recording does not provide
the two valid W20 inputs required by this experiment, so no boundary prediction
is made for it. The supervised transfer result therefore covers TX108, TX60,
and VR2 (75% of the four supervised mice).

## Leakage controls

- All feature filling, centering, scaling, target scaling, and ridge fitting
  use training mice only.
- Candidate selection is nested inside each unsupervised outer LOMO fold.
- One final candidate per area/horizon is selected on unsupervised LOMO and
  frozen before supervised outcomes are scored.
- All inputs precede the predicted after-learning window.
- The behavior feature stored by the compact cache describes the penultimate
  W20, not the terminal W20; the label and documentation preserve that fact.

## Interpretation

The strongest transferable gains are medial. That is compatible with the
hypothesis that part of mHV's cross-day population change is exposure-like.
It is not causal proof, and the repeatedly inspected three-mouse supervised
set is not a fresh confirmatory test. A new held-out cohort is required for a
confirmatory claim.

## Reproducibility

```json
{
  "alphas": [
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0
  ],
  "anchor": "last full-derived before-learning W20 population state",
  "areas": [
    "mHV",
    "aHV"
  ],
  "candidate_count": 26,
  "created_utc": "2026-07-24T05:43:43.183023+00:00",
  "feature_variants": [
    "session_only",
    "dual_full",
    "dual_full_svd",
    "dual_full_svd_behavior"
  ],
  "full_cache": "modeling/cache/full_neural_objective_b_w20.npz",
  "full_cache_sha256": "4c7e59d7e0ca98f05b45166626cfef5aebb6cb9afa7a4fbf9abd71ac7c7aa43e",
  "horizons": [
    0,
    1
  ],
  "metrics": [
    "q05",
    "median",
    "q95",
    "sd_dprime",
    "frac_leaf_selective",
    "frac_circle_selective"
  ],
  "model_version": "boundary_population_forecaster_v1",
  "numpy": "2.4.6",
  "pandas": "2.3.3",
  "prediction_target": "six area-level d-prime distribution statistics in first and second after-learning W20",
  "python": "3.12.13",
  "randomness": "none",
  "refined_cache": "modeling/cache/refined_neuron_transitions_w20.npz",
  "refined_cache_sha256": "26089d755d5f7fdca7be0f24d57a4f8238a6a19dbe92d1dabe16e358845d6e9d",
  "result_schema_version": 1,
  "run_id": "run_20260724_direct_boundary_v1",
  "scikit_learn": "1.6.0",
  "selection_rule": "minimum_cv",
  "training_cohort": "unsupervised",
  "transfer_cohort": "supervised",
  "window_pairs_per_role": 20
}
```
