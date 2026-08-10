from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import plot

def test_guide_states_input_grain_and_column_roles() -> None:
    guide = plot.guide("trajectory")

    assert list(guide.index) == ["trajectory"]
    assert guide.loc["trajectory", "input"] == "pandas DataFrame"
    assert "one row per" in guide.loc["trajectory", "grain"]
    assert "x: numeric column" in guide.loc["trajectory", "required"]
    assert "band" in guide.loc["trajectory", "optional"]


def test_numeric_categories_and_groups_keep_their_types() -> None:
    comparison_frame = pd.DataFrame({"level": [1, 2], "value": [10.0, 20.0]})
    grouped_frame = pd.DataFrame(
        {
            "step": [0.0, 1.0, 0.0, 1.0],
            "value": [1.0, 2.0, 3.0, 4.0],
            "group": [1, 1, 2, 2],
        }
    )

    points = plot.comparison(comparison_frame, x="level", y="value")
    lines = plot.trajectory(grouped_frame, x="step", y="value", by="group")
    try:
        np.testing.assert_array_equal(points.axes[0].lines[0].get_ydata(), [10.0, 20.0])
        assert [line.get_label() for line in lines.axes[0].lines] == ["1", "2"]
        np.testing.assert_array_equal(lines.axes[0].lines[1].get_ydata(), [3.0, 4.0])
    finally:
        plt.close(points)
        plt.close(lines)


def test_tabular_contract_rejects_missing_columns_nonfinite_values_and_duplicates() -> None:
    with pytest.raises(ValueError, match="missing columns.*mean"):
        plot.trajectory(pd.DataFrame({"progress": [0.0]}), x="progress", y="mean")

    with pytest.raises(ValueError, match="non-finite"):
        plot.relationship(
            pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0]}),
            x="x",
            y="y",
        )

    with pytest.raises(ValueError, match="Aggregate explicitly"):
        plot.bars(
            pd.DataFrame({"area": ["V1", "V1"], "value": [1.0, 2.0]}),
            x="area",
            y="value",
        )


def test_optional_uncertainty_may_be_unavailable() -> None:
    frame = pd.DataFrame(
        {"progress": [0.0, 1.0], "mean": [1.0, 2.0], "se": [np.nan, 0.2]}
    )
    figure = plot.trajectory(frame, x="progress", y="mean", band="se")
    try:
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [1.0, 2.0])
    finally:
        plt.close(figure)


def test_one_axis_cannot_silently_discard_facets() -> None:
    frame = pd.DataFrame(
        {
            "step": [0.0, 1.0, 0.0, 1.0],
            "value": [1.0, 2.0, 3.0, 4.0],
            "panel": ["a", "a", "b", "b"],
        }
    )
    outer, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="one axis.*2 facets"):
            plot.trajectory(frame, x="step", y="value", facet="panel", ax=ax)
    finally:
        plt.close(outer)


def test_evolution_handles_missing_windows_and_validates_options() -> None:
    values = np.array([[1.0, np.nan], [2.0, np.nan]])
    curve = plot.evolution(values, kind="curve")
    histogram = plot.evolution(values, kind="histogram")
    try:
        curve._func(0)
        histogram._func(1)
        assert "no finite observations" in [text.get_text() for text in histogram._fig.axes[0].texts]
    finally:
        curve._draw_was_started = True
        histogram._draw_was_started = True
        plt.close(curve._fig)
        plt.close(histogram._fig)

    with pytest.raises(ValueError, match="kind"):
        plot.evolution([[1.0]], kind="unknown")


def test_heatmap_and_density_use_explicit_lower_origin() -> None:
    heatmap = plot.heatmap([[1.0, 2.0], [3.0, 4.0]])
    density = plot.density(
        [[1.0, 2.0], [3.0, 4.0]],
        outlines=[np.array([[0.0, 0.0], [1.0, 1.0]])],
    )
    try:
        assert heatmap.axes[0].images[0].origin == "lower"
        assert density.axes[0].images[0].origin == "lower"
        np.testing.assert_array_equal(density.axes[0].lines[0].get_ydata(), [0.0, 1.0])
    finally:
        plt.close(heatmap)
        plt.close(density)


def test_attach_reports_missing_and_attached_views(tmp_path) -> None:
    result = tmp_path / plot.VIEWS["window_dprime"]
    result.parent.mkdir(parents=True)
    result.write_text("x,y\n1,2\n")

    class Database:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def query(self, statement: str) -> pd.DataFrame:
            self.statements.append(statement)
            return pd.DataFrame()

    database = Database()
    report = plot.attach_views(database, tmp_path)

    assert report.set_index("view").loc["window_dprime", "attached"]
    assert not report.set_index("view").loc["session_dprime", "attached"]
    assert len(database.statements) == 1
    with pytest.raises(FileNotFoundError, match="session_dprime"):
        plot.attach_views(database, tmp_path, required=True)

    assert plot.attach(database, tmp_path) is database


def test_style_and_save_options_are_declared() -> None:
    with pytest.raises(ValueError, match="scale"):
        plot.use("poster")

    figure = plot.heatmap([[1.0]])
    try:
        with pytest.raises(ValueError, match="target"):
            plot.save(figure, "unused.png", target="poster")
    finally:
        plt.close(figure)
