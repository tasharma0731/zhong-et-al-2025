import pandas as pd

from data_access import DataFrameSQL

from .inference import exact_group_permutation, exact_paired_sign_flip
from .trials import AREAS


def dprime_summary_tables(
    history,
    *,
    metrics,
    progress_bins=5,
    independent_window_pairs=None,
):
    metrics = tuple(metrics)
    if not metrics or any(not metric.isidentifier() for metric in metrics):
        raise ValueError("metrics must contain valid column names")
    missing = set(metrics) - set(history.columns)
    if missing:
        raise ValueError(f"history is missing {sorted(missing)}")
    progress_bins = int(progress_bins)
    if progress_bins < 1:
        raise ValueError("progress_bins must be positive")

    with DataFrameSQL(history=history) as sql:
        if independent_window_pairs is None:
            if "trial_bin" not in history:
                raise ValueError("disjoint histories must contain trial_bin")
            analysis_windows = sql.query("""
                SELECT *, trial_bin::INTEGER AS progress_bin
                FROM history
                ORDER BY cohort, mouse, moment, area, window_id
            """)
        else:
            analysis_windows = sql.query(f"""
                SELECT *,
                       LEAST(FLOOR(progress * {progress_bins})::INTEGER,
                             {progress_bins - 1}) + 1 AS progress_bin
                FROM history
                WHERE start_pair_id % {int(independent_window_pairs)} = 0
                ORDER BY cohort, mouse, moment, area, window_id
            """)
        sql.register("analysis_windows", analysis_windows)

        trajectory_metrics = tuple(dict.fromkeys(
            [*metrics, *[name for name in ("q05", "median", "q95") if name in history]]
        ))
        mouse_means = ",\n".join(
            f"AVG({metric}) AS {metric}" for metric in trajectory_metrics
        )
        sql.register("mouse_progress", sql.query(f"""
            SELECT mouse, cohort, moment, area, progress_bin, {mouse_means}
            FROM analysis_windows
            GROUP BY ALL
        """))
        group_means = ",\n".join(
            f"AVG({metric}) AS {metric}" for metric in trajectory_metrics
        )
        group_sems = ",\n".join(
            f"STDDEV_SAMP({metric}) / SQRT(COUNT(*)) AS {metric}_sem"
            for metric in trajectory_metrics
        )
        trajectory = sql.query(f"""
            SELECT cohort, moment, area, progress_bin,
                   COUNT(*) AS mice, {group_means}, {group_sems}
            FROM mouse_progress
            GROUP BY ALL
            ORDER BY cohort, moment, area, progress_bin
        """)

        session_means = ",\n".join(f"AVG({metric}) AS {metric}" for metric in metrics)
        session = sql.query(f"""
            SELECT behavior_session_id, recording_id, experiment, mouse,
                   cohort, moment, area, {session_means}
            FROM analysis_windows
            GROUP BY ALL
            ORDER BY cohort, mouse, moment, area
        """)
        sql.register("session_metrics", session)
        metric_names = ", ".join(metrics)
        mouse_deltas = sql.query(f"""
            WITH long_metrics AS (
                SELECT mouse, cohort, moment, area, metric, value
                FROM session_metrics
                UNPIVOT (value FOR metric IN ({metric_names}))
            )
            SELECT mouse, cohort, area, metric,
                   MAX(value) FILTER (WHERE moment = 'before') AS before,
                   MAX(value) FILTER (WHERE moment = 'after') AS after,
                   MAX(value) FILTER (WHERE moment = 'after')
                     - MAX(value) FILTER (WHERE moment = 'before') AS delta
            FROM long_metrics
            GROUP BY ALL
            HAVING before IS NOT NULL AND after IS NOT NULL
            ORDER BY area, metric, cohort, mouse
        """)
        sql.register("mouse_deltas", mouse_deltas)

        tests = []
        for area in AREAS:
            for metric in metrics:
                current = sql.query("""
                    SELECT mouse, cohort, delta
                    FROM mouse_deltas
                    WHERE area = ? AND metric = ?
                    ORDER BY cohort, mouse
                """, [area, metric])
                for cohort in ("supervised", "unsupervised"):
                    group = current.loc[current["cohort"] == cohort, "delta"]
                    result = exact_paired_sign_flip(group)
                    tests.append([
                        area, metric, "paired_sign_flip", cohort,
                        result["difference"], result["pvalue"],
                        int(result["permutations"]), int(result["n_pairs"]),
                    ])
                comparison = exact_group_permutation(
                    current["delta"],
                    current["cohort"],
                    mouse_ids=current["mouse"],
                    rewarded_label="supervised",
                    unrewarded_label="unsupervised",
                    alternative="two-sided",
                )
                tests.append([
                    area, metric, "cohort_label_permutation",
                    "supervised_minus_unsupervised", comparison["difference"],
                    comparison["pvalue"], int(comparison["permutations"]),
                    int(comparison["n_mice"]),
                ])
    inference = pd.DataFrame(
        tests,
        columns=["area", "metric", "test", "cohort", "effect", "pvalue", "permutations", "mice"],
    )
    return trajectory, analysis_windows, session, mouse_deltas, inference

