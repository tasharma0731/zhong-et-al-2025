# Zhong et al. (2025) — Neuromatch analysis

Analysis code for the Zhong et al. dataset, including:

- a Pandas/DuckDB data-access layer for the released Janelia dataset;
- the Objective A exposure-shift model;
- the Objective B boundary model; and
- analysis notebooks.

## My contribution

This is my fork of a Neuromatch Academy Computational Neuroscience team project
(Pod: Tokoloshe Black Cumin). My individual analysis and code are in
[`notebooks/tanya.ipynb`](notebooks/tanya.ipynb), covering:

- Data wrangling and neural/behavioural frame alignment
- Descriptive statistics (median, standard deviation, skewness, selectivity
  fraction) of d′ distributions across supervised and unsupervised
  conditions, computed per-session and per-window across two visual cortex
  regions (V1, mHV)
- Control-variable design, including neuron-count subsampling and
  trial-fraction binning
- The `dprime_animation_by_group.py` animation script, visualising d′
  distribution shifts across sliding trial windows, used in the team's
  final presentation

The full team pipeline below (data-access layer, GAMM-based trajectory
modelling, Objective A/B forecasting) was built by other team members and
consolidated by [Shibasis Patnaik](https://github.com/shibasis0801).

## Repository map

| Path | Contents |
|---|---|
| `code/` | Installable `zhongdb` package: Drive access, catalog tables, DuckDB construction, joins, d-prime helpers, plotting, and tests |
| `modeling/` | Objective A pipeline and Objective B boundary forecaster |
| `models/objective_a/` | Objective A fitted models and evaluation tables |
| `models/objective_b/` | Objective B fitted models, selections, predictions, report, and figures |
| `notebooks/` | Analysis notebooks |

## Install the DuckDB layer

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e './code[dev,rebuild]'
```

The data layer builds or opens one local `zhong.duckdb` file while keeping
large NumPy arrays lazy:

```python
import sql

db = sql.setup()
db.behavior().close()

recordings = db.table("recordings")
selected = db.query(
    """
    SELECT recording_id, experiment
    FROM memberships
    WHERE mouse = ? AND experiment = ?
    """,
    ["TX119", "unsup_test1"],
)

recording_id = selected.iloc[0]["recording_id"]
svd = db.load(recording_id, "reduced_neural")
```

The generated database and downloaded arrays are local working data and are
ignored by Git.

## Final models

Objective A predicts the post-exposure six-statistic population distribution
from the pre-exposure population. Its locked estimator is an area-specific,
constraint-safe mean exposure shift learned on unsupervised mice.

Objective B directly predicts the first two post-learning W20 population
states from before-learning information. Candidate family and regularization
are selected by mouse-grouped validation on unsupervised mice, then frozen for
the supervised transfer evaluation.

See [modeling/README.md](modeling/README.md),
[models/objective_a/README.md](models/objective_a/README.md), and
[models/objective_b/README.md](models/objective_b/README.md).

## Tests

```bash
pytest code/tests
pytest modeling/tests
```

## Data and evidence status

The supervised analyses are exploratory. “Frozen” describes the computational
transfer procedure; it does not make the supervised cohort a previously unseen
prospective test.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
