from __future__ import annotations

import hashlib
import inspect
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .context import GraphContext
from .display import _safe_error_text
from .types import (
    GraphError,
    Node,
    NodeError,
    Run,
    _cacheable_result,
    _fingerprint_value,
    _make_store,
)


class GraphExecution:
    def __init__(
        self,
        graph: GraphContext,
        *,
        until: str | None,
        cache: Any,
        settings: Mapping[str, Any],
    ) -> None:
        self.graph = graph
        self.nodes = graph._execution_nodes(until)
        self.settings = self._resolve_settings(settings)
        self.store = _make_store(cache, graph.name)
        self.values: dict[str, Any] = {}
        self.timings: dict[str, float] = {}
        self.order: list[str] = []
        self.fingerprints: dict[str, str | None] = {}

    def run(self) -> Run:
        for node in self.nodes:
            self._run_node(node)
        return self._record()

    def _resolve_settings(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(supplied) - set(self.graph._settings))
        if unknown:
            raise GraphError(f"unknown run settings: {', '.join(unknown)}")

        active = {
            parameter.name
            for node in self.nodes
            for parameter in node.signature.parameters.values()
            if (node.name, parameter.name) not in self.graph._connections
        }
        skipped = sorted(set(supplied) - active)
        if skipped:
            raise GraphError(
                "run settings belong to nodes outside this target: "
                + ", ".join(skipped)
            )

        settings: dict[str, Any] = {}
        for node in self.nodes:
            for parameter in node.signature.parameters.values():
                if (node.name, parameter.name) in self.graph._connections:
                    continue
                if parameter.name in supplied:
                    settings[parameter.name] = supplied[parameter.name]
                elif parameter.default is not inspect.Parameter.empty:
                    settings[parameter.name] = parameter.default
                else:
                    raise GraphError(
                        f"missing required run setting: {node.name}.{parameter.name}"
                    )
        return settings

    def _run_node(self, node: Node) -> None:
        arguments = self._arguments(node)
        fingerprint = self._fingerprint(node, arguments)
        self.fingerprints[node.name] = fingerprint

        cached = self._cached(node, fingerprint)
        if cached is not None:
            self.values.update(cached)
            self.timings[node.name] = 0.0
            self.order.append(node.name)
            return

        produced, seconds = self._execute(node, arguments)
        self.values.update(produced)
        self.timings[node.name] = seconds
        self.order.append(node.name)
        self._cache(node, fingerprint, produced)

    def _arguments(self, node: Node) -> dict[str, Any]:
        arguments = {}
        for parameter in node.signature.parameters.values():
            connection = self.graph._connections.get((node.name, parameter.name))
            arguments[parameter.name] = (
                self.settings[parameter.name]
                if connection is None
                else self.values[connection[1]]
            )
        return arguments

    def _fingerprint(
        self,
        node: Node,
        arguments: Mapping[str, Any],
    ) -> str | None:
        if self.store is None or not node.cache:
            return None
        try:
            source = inspect.getsource(node.operation)
        except (OSError, TypeError):
            return None

        parts = [source, repr(node.outputs)]
        for parameter in node.signature.parameters.values():
            connection = self.graph._connections.get((node.name, parameter.name))
            if connection is None:
                token = _fingerprint_value(arguments[parameter.name])
                if token is None:
                    return None
                parts.append(f"{parameter.name}={token}")
            else:
                upstream = self.fingerprints.get(connection[0])
                if upstream is None:
                    return None
                parts.append(f"{parameter.name}<-{upstream}")
        return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()[:32]

    def _cached(
        self,
        node: Node,
        fingerprint: str | None,
    ) -> Mapping[str, Any] | None:
        if self.store is None or fingerprint is None:
            return None
        return self.store.get({"node": node.name, "fp": fingerprint})

    def _execute(
        self,
        node: Node,
        arguments: Mapping[str, Any],
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            returned = node.operation(**arguments)
            if inspect.isawaitable(returned):
                close = getattr(returned, "close", None)
                if callable(close):
                    close()
                raise GraphError(
                    f"node {node.name!r} returned an awaitable; "
                    "async work is not supported in notebook flows"
                )
            produced = self._normalize_outputs(node, returned)
        except Exception as error:
            inputs = {
                name: self.graph._value_summary(value)
                for name, value in arguments.items()
            }
            raise NodeError(
                f"node {node.name!r} failed with inputs "
                f"{inputs!r}: {_safe_error_text(error)}",
                node_name=node.name,
                completed=self.order,
                timings=self.timings,
                inputs=inputs,
            ) from error
        return produced, time.perf_counter() - started

    def _cache(
        self,
        node: Node,
        fingerprint: str | None,
        produced: Mapping[str, Any],
    ) -> None:
        if self.store is None or fingerprint is None or not _cacheable_result(produced):
            return
        try:
            self.store.put({"node": node.name, "fp": fingerprint}, produced)
        except Exception:
            pass

    @staticmethod
    def _normalize_outputs(node: Node, returned: Any) -> dict[str, Any]:
        if len(node.outputs) == 1:
            return {node.outputs[0]: returned}
        if not isinstance(returned, Mapping):
            raise GraphError(
                f"node {node.name!r} has multiple outputs and must return a mapping"
            )
        expected = set(node.outputs)
        actual = set(returned)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise GraphError(
                f"node {node.name!r} returned the wrong ports; "
                f"missing={missing}, extra={extra}"
            )
        return {name: returned[name] for name in node.outputs}

    def _record(self) -> Run:
        executed = {node.name for node in self.nodes}
        consumed = {
            source_port
            for (target_node, _), (source_node, source_port)
            in self.graph._connections.items()
            if target_node in executed and source_node in executed
        }
        terminal_ports = tuple(
            output
            for node in self.nodes
            for output in node.outputs
            if output not in consumed
        )
        return Run(
            graph_name=self.graph.name,
            settings=MappingProxyType(dict(self.settings)),
            outputs=MappingProxyType(dict(self.values)),
            order=tuple(self.order),
            timings=MappingProxyType(dict(self.timings)),
            final_ports=self.nodes[-1].outputs,
            terminal_ports=terminal_ports,
        )
