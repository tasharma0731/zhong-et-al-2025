from __future__ import annotations

import html
import inspect
import sys
from collections.abc import Mapping
from typing import Any

import graph as graph_module
from .context import GraphContext
from .display import _safe_error_text
from .types import GraphError, NodeError, Run


class GraphWidget(GraphContext):
    @staticmethod
    def _option_matches(left: Any, right: Any) -> bool:
        if left is right:
            return True
        try:
            return bool(left == right)
        except Exception:
            return False

    @staticmethod
    def _control_widget(widgets: Any, name: str, specification: Any, default: Any):
        label = name.replace("_", " ").capitalize()
        if isinstance(specification, widgets.Widget):
            if not hasattr(specification, "value"):
                raise GraphError(
                    f"widget control {name!r} must be a value control such as "
                    "Dropdown, Checkbox, Text, IntText, or FloatText"
                )
            if hasattr(specification.style, "description_width"):
                specification.style.description_width = "initial"
            if specification.layout.width is None:
                specification.layout.width = "100%"
            specification.layout.min_width = "0"
            specification.layout.flex = "1 1 auto"
            return specification
        if isinstance(specification, range):
            specification = list(specification)
        common = {
            "description": label,
            "style": {"description_width": "initial"},
            "layout": widgets.Layout(
                width="100%", min_width="0", flex="1 1 auto"
            ),
        }
        if isinstance(specification, (list, tuple)):
            options = list(specification)
            if not options:
                raise GraphError(f"widget control {name!r} needs at least one option")
            values = [
                item[1]
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                else item
                for item in options
            ]
            value = next(
                (
                    candidate
                    for candidate in values
                    if GraphWidget._option_matches(candidate, default)
                ),
                values[0],
            )
            return widgets.Dropdown(options=options, value=value, **common)
        kind = (
            widgets.Checkbox
            if isinstance(specification, bool)
            else widgets.IntText
            if isinstance(specification, int)
            else widgets.FloatText
            if isinstance(specification, float)
            else widgets.Text
        )
        value = str(specification) if kind is widgets.Text else specification
        return kind(value=value, **common)

    def widget(
        self,
        *,
        controls: Mapping[str, Any] | None = None,
        show: str | None = None,
        auto_run: bool = False,
    ):
        """Return a node-and-port editor with targeted runs and port inspection.

        Filled input sockets receive an earlier node's output. Hollow sockets
        are editable values. Controls are placed inside the node that owns the
        port so the visual map and the executable settings stay together. The
        default runs the complete flow; ``Run to`` can stop at one node, and a
        completed run's produced ports can be inspected without rerunning.
        Set ``auto_run=True`` to execute once with the visible default values
        as the widget is built, which is useful for deterministic notebook
        cells that should show a result immediately.
        """

        widgets = self._widgets()
        if show is not None and show not in self._producer:
            raise GraphError(f"unknown output for show={show!r}")
        control_specs = dict(controls or {})
        unknown = sorted(set(control_specs) - set(self._settings))
        if unknown:
            raise GraphError(f"unknown widget controls: {', '.join(unknown)}")
        missing_controls = [
            f"{node_name}.{name}"
            for name, (node_name, parameter) in self._settings.items()
            if parameter.default is inspect.Parameter.empty
            and name not in control_specs
        ]
        if missing_controls:
            raise GraphError(
                "widget needs an explicit value control for required input ports: "
                + ", ".join(missing_controls)
            )
        defaults, control_widgets = self.setting_defaults, {}
        for name, (_, parameter) in self._settings.items():
            explicit = name in control_specs
            default = defaults[name]
            if not explicit and not isinstance(default, (bool, int, float, str)):
                continue
            specification = control_specs[name] if explicit else default
            control_widgets[name] = self._control_widget(
                widgets, name, specification, default
            )

        interactive_names = tuple(control_widgets)
        initial_settings = {
            name: control.value for name, control in control_widgets.items()
        }
        geometry = self._diagram_geometry()
        width = geometry["width"]
        height = geometry["height"]
        diagram_view = widgets.HTML(
            value=self._diagram_html(
                preview_settings=initial_settings,
                interactive_settings=interactive_names,
                canvas_only=True,
            ),
            layout=widgets.Layout(
                grid_area="canvas",
                order="0",
                width=f"{width:.0f}px",
                min_width=f"{width:.0f}px",
                height=f"{height:.0f}px",
                overflow="hidden",
            ),
        )
        diagram_view.add_class("graph-diagram-layer")
        overlay_controls = []
        for name, control in control_widgets.items():
            node_name, _ = self._settings[name]
            port_rows = geometry["rows"][node_name]["input_ports"]
            index = next(
                index
                for index, port_row in enumerate(port_rows)
                if port_row["name"] == name
            )
            x, y, _ = geometry["position"][node_name]
            output_on_row = index < len(
                geometry["rows"][node_name]["output_ports"]
            )
            control_width = (
                geometry["node_width"] * 0.56 - 14
                if output_on_row
                else geometry["node_width"] - 18
            )
            top = (
                y
                + geometry["header_height"]
                + index * geometry["port_height"]
                + 2
            )
            left = x + 9
            control.layout.grid_area = "canvas"
            control.layout.order = "1"
            control.layout.align_self = "flex-start"
            control.layout.flex = "0 0 auto"
            control.layout.width = f"{control_width:.0f}px"
            control.layout.min_width = "0"
            control.layout.max_width = f"{control_width:.0f}px"
            control.layout.height = f"{geometry['port_height'] - 4:.0f}px"
            control.layout.margin = f"{top:.0f}px 0 0 {left:.0f}px"
            control.add_class("graph-control-layer")
            overlay_controls.append(control)

        canvas = widgets.GridBox(
            [diagram_view, *overlay_controls],
            layout=widgets.Layout(
                grid_template_columns=f"{width:.0f}px",
                grid_template_rows=f"{height:.0f}px",
                grid_template_areas='"canvas"',
                flex="0 0 auto",
                width=f"{width:.0f}px",
                min_width=f"{width:.0f}px",
                height=f"{height:.0f}px",
                overflow="hidden",
            ),
        )
        canvas_scroll = widgets.Box(
            [canvas],
            layout=widgets.Layout(
                width="100%", min_width="0", overflow="auto hidden",
                padding="0 0 6px 0",
            ),
        )
        surface = widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        "<div style='font-weight:700;font-size:1rem'>"
                        f"{html.escape(self.name)}</div>"
                        "<div style='opacity:.78;font-size:.82rem;margin:.15rem 0 .4rem'>"
                        "Edit hollow input ports directly inside their nodes, then run the flow. "
                        "Filled sockets carry wired values; side-by-side branches remain "
                        "deterministic and execute sequentially in numbered order."
                        "</div>"
                    )
                ),
                canvas_scroll,
            ],
            layout=widgets.Layout(width="100%", min_width="0", grid_gap="4px"),
        )
        run_button = widgets.Button(
            description="Run flow",
            button_style="primary",
            icon="play",
            tooltip="Run to the selected step with the visible input values",
            layout=widgets.Layout(width="132px", min_width="132px"),
        )
        run_to = widgets.Dropdown(
            options=[
                ("All", None),
                *[
                    (f"{index} · {current.label}", current.name)
                    for index, current in enumerate(self.nodes, start=1)
                ],
            ],
            value=None,
            description="Run to",
            tooltip="Run all nodes, or stop after the selected node",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="190px", min_width="160px"),
        )
        inspect_port = widgets.Dropdown(
            options=[("Run first", None)],
            value=None,
            description="Inspect port",
            disabled=True,
            tooltip="Display a produced output port without rerunning any node",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="230px", min_width="190px"),
        )
        status = widgets.HTML(layout=widgets.Layout(flex="1 1 auto", min_width="0"))
        output = widgets.HTML(
            value="",
            layout=widgets.Layout(width="100%", min_width="0"),
        )

        def _set_status(message: str, *, alert: bool = False) -> None:
            status.value = (
                f"<div role='{'alert' if alert else 'status'}' "
                f"aria-live='{'assertive' if alert else 'polite'}'>{message}</div>"
            )

        _set_status("Choose input values in their nodes, then run the flow.")

        toolbar = widgets.HBox(
            [run_button, run_to, inspect_port, status],
            layout=widgets.Layout(
                width="100%",
                min_width="0",
                align_items="center",
                flex_flow="row wrap",
                grid_gap="8px",
            ),
        )
        panel = widgets.VBox(
            [toolbar, surface, output],
            layout=widgets.Layout(width="100%", min_width="0", grid_gap="6px"),
        )
        panel.last_run = None
        panel.last_error = None
        panel.run_count = 0
        panel.result_view = output
        running = False
        preview_pending = False
        inspector_syncing = False

        def _selected_settings() -> dict[str, Any]:
            return {
                name: control.value for name, control in control_widgets.items()
            }

        def _settings_for_target(
            selected: Mapping[str, Any], target: str | None
        ) -> dict[str, Any]:
            active_nodes = {current.name for current in self._execution_nodes(target)}
            return {
                name: value
                for name, value in selected.items()
                if self._settings[name][0] in active_nodes
            }

        def _reset_inspector() -> None:
            nonlocal inspector_syncing
            inspector_syncing = True
            try:
                inspect_port.options = [("Run first", None)]
                inspect_port.value = None
                inspect_port.disabled = True
            finally:
                inspector_syncing = False

        def _enable_inspector(result: Run) -> None:
            nonlocal inspector_syncing
            labels = {
                port: f"{current.label} · {port}"
                for current in self.nodes
                if current.name in result.order
                for port in current.outputs
                if port in result.outputs
            }
            inspector_syncing = True
            try:
                inspect_port.options = [
                    ("Default result", None),
                    *[(labels[port], port) for port in result.outputs],
                ]
                inspect_port.value = None
                inspect_port.disabled = False
            finally:
                inspector_syncing = False

        def _default_result(result: Run) -> Any:
            if show is not None and show in result.outputs:
                return result[show]
            return result.final

        def _display_result(result: Run, port: str | None = None) -> None:
            value = _default_result(result) if port is None else result[port]
            output.value = graph_module._result_html(value)

        def _clear_result() -> None:
            output.value = ""

        def _figure_numbers() -> set[int]:
            pyplot = sys.modules.get("matplotlib.pyplot")
            if pyplot is None:
                return set()
            try:
                return set(pyplot.get_fignums())
            except Exception:
                return set()

        def _close_new_figures(previous: set[int]) -> None:
            pyplot = sys.modules.get("matplotlib.pyplot")
            if pyplot is None:
                return
            try:
                for number in set(pyplot.get_fignums()) - previous:
                    pyplot.close(number)
            except Exception:
                pass

        def _show_failure(error: Exception, selected: Mapping[str, Any]) -> None:
            try:
                diagram_view.value = self._diagram_html(
                    preview_settings=selected,
                    interactive_settings=interactive_names,
                    error=error if isinstance(error, NodeError) else None,
                    canvas_only=True,
                )
            except Exception:
                pass

        def _preview(_change: Any = None) -> None:
            nonlocal preview_pending
            if running:
                preview_pending = True
                return
            panel.last_run = None
            panel.last_error = None
            _reset_inspector()
            _clear_result()
            try:
                selected = _selected_settings()
                diagram_view.value = self._diagram_html(
                    preview_settings=selected,
                    interactive_settings=interactive_names,
                    canvas_only=True,
                )
                _set_status("Input values changed. Run the flow to update the result.")
            except Exception as error:
                panel.last_error = error
                _set_status(html.escape(_safe_error_text(error)), alert=True)

        for control in control_widgets.values():
            control.observe(_preview, names="value")

        def inspect_changed(change: Mapping[str, Any]) -> None:
            if inspector_syncing or running or panel.last_run is None:
                return
            port = change.get("new")
            _clear_result()
            try:
                _display_result(panel.last_run, port)
            except Exception as error:
                panel.last_error = error
                _set_status(
                    "<b>Could not display the selected port:</b> "
                    f"{html.escape(_safe_error_text(error))}",
                    alert=True,
                )
                return
            panel.last_error = None
            label = "the default result" if port is None else f"port {port}"
            _set_status(
                f"Showing {html.escape(label)} from run {panel.run_count}; "
                "no nodes reran."
            )

        inspect_port.observe(inspect_changed, names="value")

        def run_clicked(_: Any) -> None:
            nonlocal running, preview_pending
            if running:
                return
            running = True
            preview_pending = False
            selected: dict[str, Any] = {}
            result: Run | None = None
            figures_before = _figure_numbers()
            disabled_states: dict[str, bool] = {}
            target = run_to.value
            run_button.disabled = True
            run_to.disabled = True
            _reset_inspector()
            for name, control in control_widgets.items():
                if hasattr(control, "disabled"):
                    disabled_states[name] = bool(control.disabled)
                    control.disabled = True
            panel.last_run = None
            panel.last_error = None
            _clear_result()
            try:
                selected = _selected_settings()
                active_selected = _settings_for_target(selected, target)
                map_error: Exception | None = None
                try:
                    diagram_view.value = self._diagram_html(
                        preview_settings=selected,
                        interactive_settings=interactive_names,
                        canvas_only=True,
                    )
                except Exception as error:
                    map_error = error
                _set_status("Running…")
                run_error: Exception | None = None
                try:
                    result = self.run(until=target, **active_selected)
                except Exception as error:
                    run_error = error
                if run_error is not None:
                    panel.last_error = run_error
                    _show_failure(run_error, selected)
                    _set_status(
                        f"<b>Could not run:</b> "
                        f"{html.escape(_safe_error_text(run_error))}",
                        alert=True,
                    )
                    return

                assert result is not None
                panel.last_run = result
                _enable_inspector(result)
                panel.run_count += 1
                try:
                    diagram_view.value = self._diagram_html(
                        result,
                        interactive_settings=interactive_names,
                        canvas_only=True,
                    )
                    map_error = None
                except Exception as error:
                    map_error = error

                result_error: Exception | None = None
                try:
                    _display_result(result)
                except Exception as error:
                    result_error = error
                if result_error is not None:
                    panel.last_error = result_error
                    map_note = (
                        " The node map was also unavailable."
                        if map_error is not None
                        else ""
                    )
                    _set_status(
                        "<b>Run completed, but the result view failed:</b> "
                        f"{html.escape(_safe_error_text(result_error))}"
                        f"{map_note}",
                        alert=True,
                    )
                    return

                if map_error is not None:
                    panel.last_error = map_error
                    _set_status(
                        f"Run {panel.run_count} completed and the result is shown, "
                        "but the node map is unavailable: "
                        f"{html.escape(_safe_error_text(map_error))}",
                        alert=True,
                    )
                    return

                panel.last_error = None
                chosen = ", ".join(
                    f"{name.replace('_', ' ')}={self._format_setting_value(value)}"
                    for name, value in active_selected.items()
                )
                detail = f" Inputs: {html.escape(chosen)}." if chosen else ""
                target_detail = (
                    ""
                    if target is None
                    else f" Stopped after {html.escape(self._nodes_by_name[target].label)}."
                )
                _set_status(
                    f"Run {panel.run_count} complete. Ran {len(result.order)} nodes in "
                    f"{result.seconds:.3f} seconds."
                    f"{target_detail}"
                    f"{detail}"
                )
            except Exception as error:
                panel.last_run = None
                panel.last_error = error
                _show_failure(error, selected)
                _set_status(
                    f"<b>Could not run:</b> "
                    f"{html.escape(_safe_error_text(error))}",
                    alert=True,
                )
            finally:
                _close_new_figures(figures_before)
                for name, disabled in disabled_states.items():
                    control_widgets[name].disabled = disabled
                run_button.disabled = False
                run_to.disabled = False
                running = False
                if preview_pending:
                    preview_pending = False
                    _preview()

        run_button.on_click(run_clicked)
        if auto_run:
            run_clicked(None)
        return panel
