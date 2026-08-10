from __future__ import annotations

import inspect
import pprint
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from . import display as graph_display
from .display import _dict_html, _safe_error_text
from .types import (
    GraphError,
    Node,
    NodeError,
    Run,
    knob,
    node,
    use_drive_cache,
)
from .diagram import GraphDiagram
from .execution import GraphExecution
from .widget import GraphWidget


_RESULT_TEXT_LIMIT = 12_000
_RESULT_MARKUP_LIMIT = 1_000_000
_RESULT_PNG_LIMIT = 8 * 1024 * 1024


def _result_html(value: Any) -> str:
    graph_display._RESULT_TEXT_LIMIT = _RESULT_TEXT_LIMIT
    graph_display._RESULT_MARKUP_LIMIT = _RESULT_MARKUP_LIMIT
    graph_display._RESULT_PNG_LIMIT = _RESULT_PNG_LIMIT
    return graph_display._result_html(value)


class Graph(GraphDiagram, GraphWidget):
    def __init__(self, name: str, *nodes: Node):
        if not isinstance(name, str) or not name.strip():
            raise GraphError("graph name must be non-empty")
        if not nodes:
            raise GraphError("a graph needs at least one node")
        self.name = name.strip()
        self.nodes = tuple(nodes)
        self._nodes_by_name: dict[str, Node] = {}
        self._producer: dict[str, str] = {}
        self._connections: dict[tuple[str, str], tuple[str, str]] = {}
        self._settings: dict[str, tuple[str, inspect.Parameter]] = {}
        self._build()

    def _build(self) -> None:
        for current in self.nodes:
            if not isinstance(current, Node):
                if (
                    type(current).__name__ == "Node"
                    and type(current).__module__ in {"graph", Node.__module__}
                ):
                    raise GraphError(
                        "this node was created by an earlier graph module instance; "
                        "rerun its @graph.node definition cell after the setup cell"
                    )
                raise GraphError("Graph accepts only functions decorated with @node")
            if current.name in self._nodes_by_name:
                raise GraphError(f"duplicate node name: {current.name!r}")
            self._nodes_by_name[current.name] = current

            for parameter in current.signature.parameters.values():
                producer_name = self._producer.get(parameter.name)
                if producer_name is not None:
                    self._connections[(current.name, parameter.name)] = (
                        producer_name,
                        parameter.name,
                    )
                    continue
                previous = self._settings.get(parameter.name)
                if previous is not None:
                    raise GraphError(
                        f"setting {parameter.name!r} appears in both "
                        f"{previous[0]!r} and {current.name!r}; rename one for clarity"
                    )
                self._settings[parameter.name] = (current.name, parameter)

            for output in current.outputs:
                producer = self._producer.get(output)
                if producer is not None:
                    raise GraphError(
                        f"output port {output!r} is produced by both "
                        f"{producer!r} and {current.name!r}"
                    )
                earlier_setting = self._settings.get(output)
                if earlier_setting is not None:
                    raise GraphError(
                        f"output port {output!r} is declared after it was used as "
                        f"setting {earlier_setting[0]}.{output}; move its producer earlier"
                    )
                self._producer[output] = current.name

    @property
    def setting_defaults(self) -> Mapping[str, Any]:
        defaults = {}
        for name, (_, parameter) in self._settings.items():
            defaults[name] = (
                None
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
        return MappingProxyType(defaults)

    def validate(self, **settings: Any) -> "Graph":
        unknown = sorted(set(settings) - set(self._settings))
        if unknown:
            raise GraphError(f"unknown run settings: {', '.join(unknown)}")
        missing = [
            f"{node_name}.{name}"
            for name, (node_name, parameter) in self._settings.items()
            if parameter.default is inspect.Parameter.empty and name not in settings
        ]
        if missing:
            joined = ", ".join(missing)
            raise GraphError(
                "required run settings have no defaults: "
                f"{joined}. Supply them to run() or add function defaults."
            )
        return self

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description for inspection or documentation."""

        node_rows = []
        for current in self.nodes:
            inputs = []
            settings = []
            input_ports = []
            for parameter in current.signature.parameters.values():
                connection = self._connections.get((current.name, parameter.name))
                if connection is None:
                    settings.append(parameter.name)
                else:
                    inputs.append(parameter.name)
                input_ports.append(
                    {
                        "name": parameter.name,
                        "endpoint": f"{current.name}.{parameter.name}",
                        "type": self._input_type(current, parameter),
                        "connected": connection is not None,
                        "source": (
                            None
                            if connection is None
                            else f"{connection[0]}.{connection[1]}"
                        ),
                        "editable": connection is None,
                        "required": parameter.default is inspect.Parameter.empty,
                    }
                )
            node_rows.append(
                {
                    "name": current.name,
                    "label": current.label,
                    "inputs": inputs,
                    "settings": settings,
                    "outputs": list(current.outputs),
                    "input_ports": input_ports,
                    "output_ports": [
                        {
                            "name": output,
                            "endpoint": f"{current.name}.{output}",
                            "type": self._output_type(current, output),
                            "connected": any(
                                source == (current.name, output)
                                for source in self._connections.values()
                            ),
                        }
                        for output in current.outputs
                    ],
                }
            )
        edge_rows = [
            {
                "from": f"{source_node}.{source_port}",
                "to": f"{target_node}.{target_port}",
            }
            for (target_node, target_port), (source_node, source_port)
            in self._connections.items()
        ]
        return {
            "name": self.name,
            "nodes": node_rows,
            "connections": edge_rows,
            "settings": list(self._settings),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.describe()

    def __repr__(self) -> str:
        return f"Graph({pprint.pformat(self.to_dict(), width=100, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())

    def _stop_index(self, until: str | None) -> int:
        if until is None:
            return len(self.nodes) - 1
        matches = {
            index
            for index, current in enumerate(self.nodes)
            if until == current.name or until in current.outputs
        }
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            raise GraphError(
                f"ambiguous until={until!r}; it names different nodes or outputs"
            )
        raise GraphError(f"unknown node or output for until={until!r}")

    def _execution_nodes(self, until: str | None) -> tuple[Node, ...]:
        if until is None:
            return self.nodes
        target = self.nodes[self._stop_index(until)]
        required = {target.name}
        changed = True
        while changed:
            changed = False
            for (target_node, _), (source_node, _) in self._connections.items():
                if target_node in required and source_node not in required:
                    required.add(source_node)
                    changed = True
        return tuple(current for current in self.nodes if current.name in required)

    def run(self, *, until: str | None = None, cache: Any = None, **settings: Any) -> Run:
        return GraphExecution(self, until=until, cache=cache, settings=settings).run()

    def run_many(
        self,
        variations: Sequence[Mapping[str, Any]],
        *,
        until: str | None = None,
    ) -> tuple[Run, ...]:
        """Run an explicit ordered list of variations.

        The method intentionally does not create a Cartesian parameter grid.
        """

        if isinstance(variations, (str, bytes, Mapping)):
            raise TypeError("variations must be an ordered sequence of mappings")
        runs = []
        for index, variation in enumerate(variations):
            if not isinstance(variation, Mapping):
                raise TypeError(f"variation {index} is not a mapping")
            try:
                runs.append(self.run(until=until, **dict(variation)))
            except Exception as error:
                raise GraphError(
                    f"variation {index} failed: {_safe_error_text(error)}"
                ) from error
        return tuple(runs)


__all__ = [
    "Graph",
    "GraphError",
    "Node",
    "NodeError",
    "Run",
    "knob",
    "node",
    "use_drive_cache",
]
