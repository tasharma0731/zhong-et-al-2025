from __future__ import annotations

import html
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from .context import GraphContext
from .types import Node, NodeError, Run, value_summary


class GraphDiagram(GraphContext):
    @staticmethod
    def _widgets():
        try:
            import ipywidgets as widgets
        except ImportError as error:
            raise ImportError(
                "The notebook view needs ipywidgets. Install it with "
                "`python -m pip install ipywidgets`."
            ) from error
        return widgets

    @staticmethod
    def _type_label(annotation: Any, default: Any = inspect.Parameter.empty) -> str:
        if annotation is inspect.Parameter.empty:
            if default is inspect.Parameter.empty or default is None:
                return "value"
            annotation = type(default)
        if annotation is Any:
            return "value"
        if isinstance(annotation, str):
            label = annotation
        elif isinstance(annotation, type):
            label = annotation.__name__
        else:
            label = str(annotation).replace("typing.", "")
        return label if len(label) <= 22 else label[:21] + "…"

    @staticmethod
    def _type_colour(type_label: str) -> str:
        lowered = type_label.lower()
        if "bool" in lowered:
            return "#d84a4a"
        if "float" in lowered:
            return "#d8b936"
        if "int" in lowered:
            return "#35b99a"
        if "str" in lowered or "text" in lowered:
            return "#c56bd6"
        if any(word in lowered for word in ("dict", "map", "list", "tuple", "array")):
            return "#2f9fd6"
        return "#5aa7d6"

    def _input_type(self, current: Node, parameter: inspect.Parameter) -> str:
        return self._type_label(parameter.annotation, parameter.default)

    def _output_type(self, current: Node, port: str) -> str:
        annotation = current.signature.return_annotation
        if len(current.outputs) != 1:
            annotation = inspect.Signature.empty
        return self._type_label(annotation)

    @staticmethod
    def _value_summary(value: Any) -> str:
        return value_summary(value)

    @staticmethod
    def _format_setting_value(value: Any) -> str:
        text = value_summary(value)
        return text if len(text) <= 32 else text[:31] + "…"

    @staticmethod
    def _fit_port_label(label: str, *, shared_row: bool) -> str:
        limit = 22 if shared_row else 38
        return label if len(label) <= limit else label[: limit - 1] + "…"

    def _diagram_geometry(self) -> dict[str, Any]:
        """Return the one compact layout shared by SVG wires and UI controls."""

        node_width = 256.0
        header_height = 42.0
        port_height = 30.0
        column_gap = 88.0
        row_gap = 28.0

        description = self.describe()
        rows = {row["name"]: row for row in description["nodes"]}
        depth: dict[str, int] = {}
        for current in self.nodes:
            sources = [
                source_node
                for (target_node, _), (source_node, _) in self._connections.items()
                if target_node == current.name
            ]
            depth[current.name] = (
                0 if not sources else 1 + max(depth[source] for source in sources)
            )
        max_depth = max(depth.values()) if depth else 0
        bypass_candidates = sum(
            depth[target_node] - depth[source_node] > 1
            for (target_node, _), (source_node, _) in self._connections.items()
        )
        margin = max(18.0, 12.0 + 6.0 * bypass_candidates)

        columns: dict[int, list[str]] = {}
        for current in self.nodes:
            columns.setdefault(depth[current.name], []).append(current.name)

        position: dict[str, tuple[float, float, float]] = {}
        canvas_bottom = margin
        for column in range(max_depth + 1):
            x = margin + column * (node_width + column_gap)
            y = margin
            for name in columns.get(column, []):
                row = rows[name]
                body_rows = max(
                    len(row["input_ports"]), len(row["output_ports"]), 1
                )
                height = header_height + body_rows * port_height + 14
                position[name] = (x, y, height)
                y += height + row_gap
            canvas_bottom = max(canvas_bottom, y)

        width = margin * 2 + (max_depth + 1) * node_width + max_depth * column_gap
        height = canvas_bottom - row_gap + margin
        return {
            "description": description,
            "rows": rows,
            "position": position,
            "node_width": node_width,
            "header_height": header_height,
            "port_height": port_height,
            "margin": margin,
            "bypass_candidates": bypass_candidates,
            "width": width,
            "height": height,
        }

    @staticmethod
    def _wire_path(
        source_node: str,
        target_node: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        geometry: Mapping[str, Any],
        lane_index: int = 0,
    ) -> tuple[str, str, float | None]:
        """Route a wire around any node boxes between its endpoints."""

        node_width = float(geometry["node_width"])
        positions = geometry["position"]
        intermediates = [
            (x, y, height)
            for name, (x, y, height) in positions.items()
            if name not in {source_node, target_node}
            and x1 < x + node_width
            and x < x2
        ]
        clearance = 4.0
        low, high = sorted((y1, y2))
        blocked = any(
            low <= y + height + clearance and high >= y - clearance
            for _, y, height in intermediates
        )
        if not blocked:
            bend = max(40.0, (x2 - x1) * 0.5)
            return (
                f"M{x1:.1f},{y1:.1f} C{x1 + bend:.1f},{y1:.1f} "
                f"{x2 - bend:.1f},{y2:.1f} {x2:.1f},{y2:.1f}",
                "direct",
                None,
            )

        lane_y = max(
            4.0,
            min(y for _, y, _ in intermediates) - 12.0 - lane_index * 6.0,
        )
        before = min(x for x, _, _ in intermediates) - 12.0
        after = max(x + node_width for x, _, _ in intermediates) + 12.0
        first_turn = (x1 + before) / 2
        second_turn = (after + x2) / 2
        return (
            f"M{x1:.1f},{y1:.1f} C{first_turn:.1f},{y1:.1f} "
            f"{first_turn:.1f},{lane_y:.1f} {before:.1f},{lane_y:.1f} "
            f"H{after:.1f} C{second_turn:.1f},{lane_y:.1f} "
            f"{second_turn:.1f},{y2:.1f} {x2:.1f},{y2:.1f}",
            "bypass",
            lane_y,
        )

    def _diagram_html(
        self,
        run: Run | None = None,
        *,
        preview_settings: Mapping[str, Any] | None = None,
        interactive_settings: Sequence[str] = (),
        error: NodeError | None = None,
        canvas_only: bool = False,
    ) -> str:
        """Render the flow as an SVG node canvas: ports, wires, and values.

        Nodes sit in columns by dependency depth. Input ports are on the left,
        output ports on the right, curved wires connect matching names, and each
        node shows its current setting values. ``preview_settings`` shows pending
        control values before a run; ``run`` shows the values and timings used.
        """

        geometry = self._diagram_geometry()
        description = geometry["description"]
        rows = geometry["rows"]
        position = geometry["position"]
        node_width = geometry["node_width"]
        header_height = geometry["header_height"]
        port_height = geometry["port_height"]
        width = geometry["width"]
        height = geometry["height"]
        port_radius = 5.0
        interactive = set(interactive_settings)

        completed = (
            set(run.order)
            if run is not None
            else set(error.completed)
            if error is not None
            else set()
        )
        timings = (
            run.timings
            if run is not None
            else error.timings
            if error is not None
            else {}
        )
        if run is not None:
            values: dict[str, Any] = dict(run.settings)
        elif preview_settings is not None:
            values = {**dict(self.setting_defaults), **dict(preview_settings)}
        else:
            values = dict(self.setting_defaults)
        output_values = {} if run is None else dict(run.outputs)

        def in_xy(name: str, index: int) -> tuple[float, float]:
            x, y, _ = position[name]
            return x, y + header_height + index * port_height + port_height / 2

        def out_xy(name: str, index: int) -> tuple[float, float]:
            x, y, _ = position[name]
            return x + node_width, y + header_height + index * port_height + port_height / 2

        wires: list[str] = []
        bypass_lane_index = 0
        for (target_node, target_port), (source_node, source_port) in self._connections.items():
            source_rows = rows[source_node]["output_ports"]
            target_rows = rows[target_node]["input_ports"]
            x1, y1 = out_xy(
                source_node,
                next(
                    index
                    for index, item in enumerate(source_rows)
                    if item["name"] == source_port
                ),
            )
            x2, y2 = in_xy(
                target_node,
                next(
                    index
                    for index, item in enumerate(target_rows)
                    if item["name"] == target_port
                ),
            )
            wire_path, route, lane_y = self._wire_path(
                source_node,
                target_node,
                x1,
                y1,
                x2,
                y2,
                geometry,
                bypass_lane_index,
            )
            if route == "bypass":
                bypass_lane_index += 1
            source_type = next(
                item["type"] for item in source_rows if item["name"] == source_port
            )
            colour = self._type_colour(source_type)
            wire_label = (
                f"{source_node}.{source_port} to {target_node}.{target_port}"
            )
            lane_attribute = (
                f"data-lane-y='{lane_y:.1f}' " if lane_y is not None else ""
            )
            wires.append(
                f"<path d='{wire_path}' fill='none' "
                f"stroke='{colour}' stroke-width='2' stroke-opacity='0.9' "
                "class='graph-wire' "
                f"aria-label='{html.escape(wire_label, quote=True)}' "
                f"data-route='{route}' "
                f"{lane_attribute}"
                f"data-source='{html.escape(source_node)}.{html.escape(source_port)}' "
                f"data-target='{html.escape(target_node)}.{html.escape(target_port)}'>"
                f"<title>{html.escape(wire_label)}</title></path>"
            )

        parts: list[str] = []
        for order_number, current in enumerate(self.nodes, start=1):
            row = rows[current.name]
            x, y, box_h = position[current.name]
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{node_width:.0f}' height='{box_h:.1f}' "
                "rx='9' fill='currentColor' fill-opacity='0.04' stroke='currentColor' "
                "stroke-opacity='0.35'/>"
            )
            parts.append(
                f"<rect x='{x + 1:.1f}' y='{y + 1:.1f}' width='{node_width - 2:.0f}' "
                f"height='{header_height - 1:.1f}' rx='8' fill='#2f9fd6' fill-opacity='0.18'/>"
            )
            parts.append(
                f"<path d='M{x:.1f},{y + header_height:.1f} h{node_width:.0f}' "
                "stroke='currentColor' stroke-opacity='0.18'/>"
            )
            parts.append(
                f"<text x='{x + 12:.1f}' y='{y + 19:.1f}' font-size='13' font-weight='600' "
                f"fill='currentColor'>{html.escape(row['label'])}</text>"
            )
            parts.append(
                f"<circle cx='{x + node_width - 17:.1f}' cy='{y + 17:.1f}' r='10' "
                "fill='currentColor' fill-opacity='0.12' stroke='currentColor' "
                "stroke-opacity='0.32'/>"
                f"<text x='{x + node_width - 17:.1f}' y='{y + 20.5:.1f}' "
                "text-anchor='middle' font-size='10' font-weight='700' "
                f"fill='currentColor'>{order_number}</text>"
            )
            if error is not None and current.name == error.node_name:
                state, colour = "failed", "fill='#d84a4a'"
            elif current.name in completed:
                milliseconds = timings.get(current.name, 0.0) * 1000
                state, colour = f"done · {milliseconds:.1f} ms", "fill='hsl(145,58%,46%)'"
            else:
                state, colour = "ready", "fill='currentColor' fill-opacity='0.55'"
            parts.append(
                f"<text x='{x + 12:.1f}' y='{y + 34:.1f}' font-size='10' "
                f"{colour}>{html.escape(state)}</text>"
            )
            for index, port_row in enumerate(row["input_ports"]):
                port = port_row["name"]
                px, py = in_xy(current.name, index)
                colour = self._type_colour(port_row["type"])
                connected = port_row["connected"]
                if not connected and port in interactive:
                    label = ""
                elif connected:
                    label = port
                else:
                    required = port_row["required"] and port not in values
                    value = (
                        "required"
                        if required
                        else self._format_setting_value(values.get(port))
                    )
                    label = f"{port} = {value}"
                parts.append(
                    f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{port_radius}' "
                    f"fill='{colour if connected else 'none'}' stroke='{colour}' stroke-width='2' "
                    f"data-endpoint='{html.escape(current.name)}.{html.escape(port)}' "
                    f"data-direction='input' data-type='{html.escape(port_row['type'])}' "
                    f"data-connected='{str(connected).lower()}'>"
                    f"<title>{html.escape(port)} input · {html.escape(port_row['type'])}</title>"
                    "</circle>"
                )
                if label:
                    visible_label = self._fit_port_label(
                        label,
                        shared_row=index < len(row["output_ports"]),
                    )
                    parts.append(
                        f"<text x='{px + 9:.1f}' y='{py + 3.5:.1f}' font-size='11' "
                        "fill='currentColor' fill-opacity='0.88'>"
                        f"<title>{html.escape(label)}</title>"
                        f"{html.escape(visible_label)}</text>"
                    )
            for index, port_row in enumerate(row["output_ports"]):
                port = port_row["name"]
                px, py = out_xy(current.name, index)
                colour = self._type_colour(port_row["type"])
                value = output_values.get(port, inspect.Parameter.empty)
                connected = port_row["connected"]
                role = (
                    ""
                    if connected
                    else " · result"
                    if current is self.nodes[-1]
                    else " · unused"
                )
                label = (
                    port
                    if value is inspect.Parameter.empty
                    else f"{port} = {self._format_setting_value(value)}"
                )
                label += role
                visible_label = self._fit_port_label(
                    label,
                    shared_row=index < len(row["input_ports"]),
                )
                parts.append(
                    f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{port_radius}' "
                    f"fill='{colour if connected else 'none'}' stroke='{colour}' stroke-width='2' "
                    f"data-endpoint='{html.escape(current.name)}.{html.escape(port)}' "
                    f"data-direction='output' data-type='{html.escape(port_row['type'])}' "
                    f"data-connected='{str(port_row['connected']).lower()}'>"
                    f"<title>{html.escape(port)} output · {html.escape(port_row['type'])}</title>"
                    "</circle>"
                    f"<text x='{px - 9:.1f}' y='{py + 3.5:.1f}' font-size='11' "
                    f"text-anchor='end' fill='currentColor' fill-opacity='0.85'>"
                    f"<title>{html.escape(label)}</title>"
                    f"{html.escape(visible_label)}</text>"
                )

        if error is not None:
            caption = f"stopped at {error.node_name} · completed steps stay visible"
        elif run is not None:
            caption = (
                f"ran {len(completed)} node(s) in {run.seconds:.3f}s · "
                "values below are this run"
            )
        else:
            caption = (
                "inputs left · outputs right · hollow sockets are editable or unused · "
                "side-by-side branches still run sequentially in numbered order"
            )
        order_text = " then ".join(current.name for current in self.nodes)
        connection_text = ", ".join(
            f"{edge['from']} to {edge['to']}" for edge in description["connections"]
        ) or "no wires"
        setting_text = ", ".join(
            f"{name}={self._format_setting_value(values.get(name))}"
            for name in self._settings
        ) or "no editable inputs"
        accessible = (
            f"{self.name}. Node order: {order_text}. "
            f"Connections: {connection_text}. Current input values: {setting_text}."
        )
        svg = (
            f"<svg xmlns='http://www.w3.org/2000/svg' "
            f"viewBox='0 0 {width:.0f} {height:.0f}' width='{width:.0f}' "
            f"height='{height:.0f}' role='img' aria-label='{html.escape(accessible, quote=True)}' "
            "style='height:auto;font-family:inherit;display:block'>"
            f"<title>{html.escape(self.name)} flow</title>"
            f"<desc>{html.escape(accessible)}</desc>"
            "<style>.graph-wire:hover{stroke-width:4}</style>"
            + "".join(wires)
            + "".join(parts)
            + "</svg>"
        )
        if canvas_only:
            return (
                "<style>"
                ".graph-diagram-layer{position:relative;z-index:0}"
                ".graph-control-layer{position:relative;z-index:2}"
                "</style>"
                + svg
            )
        return (
            "<div style='margin:.25rem 0 .75rem'>"
            f"<div style='font-weight:600'>{html.escape(self.name)}</div>"
            f"<div style='opacity:.6;font-size:.8rem;margin:.1rem 0 .4rem'>"
            f"{html.escape(caption)}</div>"
            f"<div style='overflow-x:auto'>{svg}</div></div>"
        )

    def diagram(self, run: Run | None = None):
        widgets = self._widgets()
        return widgets.HTML(value=self._diagram_html(run))
