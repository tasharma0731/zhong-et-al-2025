# Actual-value accuracy

This is a post-hoc, descriptive summary of the frozen `supervised_test` predictions.
One evaluation unit is one predicted **area-level population statistic** for
one mouse, area, and forecast horizon. It is not a neuron-level accuracy.

The primary table pools only q05, median, and q95 because these are all
d-prime location statistics. `within_0_5_pct` is the percentage of those
individual statistic predictions with absolute error no greater than 0.5
d-prime. D-prime SD is reported in a second four-statistic summary, and the
two selectivity fractions are kept in their native 0–1 units.

The supervised split contains three eligible mice (TX108, TX60, and VR2).
TX109 is excluded by the model contract because it lacks two valid
before-learning W20 windows.

## q05 / median / q95 headline

| area | horizon | model_label | value_predictions | mae | rmse | within_0_5_pct |
| --- | --- | --- | --- | --- | --- | --- |
| aHV | W1 | Last-before W20 persistence | 9.0 | 0.253 | 0.302 | 88.9 |
| aHV | W1 | Direct boundary forecaster | 9.0 | 0.198 | 0.250 | 100.0 |
| aHV | W2 | Last-before W20 persistence | 9.0 | 0.483 | 0.632 | 55.6 |
| aHV | W2 | Direct boundary forecaster | 9.0 | 0.435 | 0.576 | 66.7 |
| aHV | both | Last-before W20 persistence | 18.0 | 0.368 | 0.495 | 72.2 |
| aHV | both | Direct boundary forecaster | 18.0 | 0.317 | 0.444 | 83.3 |
| mHV | W1 | Last-before W20 persistence | 9.0 | 0.427 | 0.485 | 66.7 |
| mHV | W1 | Direct boundary forecaster | 9.0 | 0.155 | 0.174 | 100.0 |
| mHV | W2 | Last-before W20 persistence | 9.0 | 0.620 | 0.759 | 44.4 |
| mHV | W2 | Direct boundary forecaster | 9.0 | 0.258 | 0.360 | 77.8 |
| mHV | both | Last-before W20 persistence | 18.0 | 0.523 | 0.637 | 55.6 |
| mHV | both | Direct boundary forecaster | 18.0 | 0.207 | 0.283 | 88.9 |
| overall | W1 | Last-before W20 persistence | 18.0 | 0.340 | 0.404 | 77.8 |
| overall | W1 | Direct boundary forecaster | 18.0 | 0.177 | 0.215 | 100.0 |
| overall | W2 | Last-before W20 persistence | 18.0 | 0.552 | 0.698 | 50.0 |
| overall | W2 | Direct boundary forecaster | 18.0 | 0.347 | 0.480 | 72.2 |
| overall | both | Last-before W20 persistence | 36.0 | 0.446 | 0.571 | 63.9 |
| overall | both | Direct boundary forecaster | 36.0 | 0.262 | 0.372 | 86.1 |

## q05 / median / q95 / d-prime SD

| area | horizon | model_label | value_predictions | mae | rmse | within_0_5_pct |
| --- | --- | --- | --- | --- | --- | --- |
| aHV | W1 | Last-before W20 persistence | 12.0 | 0.220 | 0.269 | 91.7 |
| aHV | W1 | Direct boundary forecaster | 12.0 | 0.172 | 0.222 | 100.0 |
| aHV | W2 | Last-before W20 persistence | 12.0 | 0.453 | 0.585 | 58.3 |
| aHV | W2 | Direct boundary forecaster | 12.0 | 0.398 | 0.528 | 66.7 |
| aHV | both | Last-before W20 persistence | 24.0 | 0.337 | 0.455 | 75.0 |
| aHV | both | Direct boundary forecaster | 24.0 | 0.285 | 0.405 | 83.3 |
| mHV | W1 | Last-before W20 persistence | 12.0 | 0.441 | 0.486 | 66.7 |
| mHV | W1 | Direct boundary forecaster | 12.0 | 0.158 | 0.175 | 100.0 |
| mHV | W2 | Last-before W20 persistence | 12.0 | 0.642 | 0.751 | 33.3 |
| mHV | W2 | Direct boundary forecaster | 12.0 | 0.269 | 0.353 | 83.3 |
| mHV | both | Last-before W20 persistence | 24.0 | 0.542 | 0.632 | 50.0 |
| mHV | both | Direct boundary forecaster | 24.0 | 0.213 | 0.279 | 91.7 |
| overall | W1 | Last-before W20 persistence | 24.0 | 0.330 | 0.393 | 79.2 |
| overall | W1 | Direct boundary forecaster | 24.0 | 0.165 | 0.200 | 100.0 |
| overall | W2 | Last-before W20 persistence | 24.0 | 0.548 | 0.673 | 45.8 |
| overall | W2 | Direct boundary forecaster | 24.0 | 0.334 | 0.450 | 75.0 |
| overall | both | Last-before W20 persistence | 48.0 | 0.439 | 0.551 | 62.5 |
| overall | both | Direct boundary forecaster | 48.0 | 0.249 | 0.348 | 87.5 |

## Leaf- and circle-selective fractions

| area | horizon | model_label | value_predictions | mae | rmse | within_0_05_fraction_pct | within_0_10_fraction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aHV | W1 | Last-before W20 persistence | 6.0 | 0.160 | 0.177 | 0.0 | 16.7 |
| aHV | W1 | Direct boundary forecaster | 6.0 | 0.113 | 0.136 | 16.7 | 50.0 |
| aHV | W2 | Last-before W20 persistence | 6.0 | 0.156 | 0.189 | 16.7 | 50.0 |
| aHV | W2 | Direct boundary forecaster | 6.0 | 0.127 | 0.152 | 33.3 | 50.0 |
| aHV | both | Last-before W20 persistence | 12.0 | 0.158 | 0.183 | 8.3 | 33.3 |
| aHV | both | Direct boundary forecaster | 12.0 | 0.120 | 0.144 | 25.0 | 50.0 |
| mHV | W1 | Last-before W20 persistence | 6.0 | 0.111 | 0.130 | 33.3 | 50.0 |
| mHV | W1 | Direct boundary forecaster | 6.0 | 0.080 | 0.088 | 16.7 | 66.7 |
| mHV | W2 | Last-before W20 persistence | 6.0 | 0.076 | 0.096 | 50.0 | 50.0 |
| mHV | W2 | Direct boundary forecaster | 6.0 | 0.032 | 0.037 | 83.3 | 100.0 |
| mHV | both | Last-before W20 persistence | 12.0 | 0.093 | 0.114 | 41.7 | 50.0 |
| mHV | both | Direct boundary forecaster | 12.0 | 0.056 | 0.067 | 50.0 | 83.3 |
| overall | W1 | Last-before W20 persistence | 12.0 | 0.135 | 0.155 | 16.7 | 33.3 |
| overall | W1 | Direct boundary forecaster | 12.0 | 0.096 | 0.114 | 16.7 | 58.3 |
| overall | W2 | Last-before W20 persistence | 12.0 | 0.116 | 0.150 | 33.3 | 50.0 |
| overall | W2 | Direct boundary forecaster | 12.0 | 0.079 | 0.111 | 58.3 | 75.0 |
| overall | both | Last-before W20 persistence | 24.0 | 0.126 | 0.153 | 25.0 | 41.7 |
| overall | both | Direct boundary forecaster | 24.0 | 0.088 | 0.113 | 37.5 | 66.7 |

## Interpretation

`mae` and `rmse` are in d-prime units for the three-statistic and
four-statistic summaries. `robust_scale_nrmse` divides each output error by
the unsupervised-training robust scale stored with the frozen predictions;
it is dimensionless but is **not a percentage**. Selective-fraction MAE is a
fraction: for example, 0.08 means 8 percentage points.

All aggregate percentages are descriptive and have coarse resolution because
there are only three eligible supervised mice. An area × horizon cell has
9 q05/median/q95 value predictions, so one prediction changes its percentage
by 11.1 points.
