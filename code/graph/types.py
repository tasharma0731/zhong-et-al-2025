from __future__ import annotations

import hashlib
import inspect
import pprint
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .display import _dict_html, _is_matplotlib_figure, _safe_error_text


class GraphError(ValueError):
    """Raised when a graph or run setting is incomplete or ambiguous."""


class NodeError(RuntimeError):
    """Raised when one node fails while a graph is running.

    The compact attributes are useful in notebooks: they identify the failed
    step without printing large arrays or hiding work completed beforehand.
    """

    def __init__(
        self,
        message: str,
        *,
        node_name: str,
        completed: Sequence[str] = (),
        timings: Mapping[str, float] | None = None,
        inputs: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.node_name = node_name
        self.completed = tuple(completed)
        self.timings = MappingProxyType(dict(timings or {}))
        self.inputs = MappingProxyType(dict(inputs or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": _safe_error_text(self),
            "node_name": self.node_name,
            "completed": self.completed,
            "timings_seconds": dict(self.timings),
            "inputs": dict(self.inputs),
        }

    def __repr__(self) -> str:
        return f"NodeError({pprint.pformat(self.to_dict(), width=88, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


def _public_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and len(value) <= 120:
        return value
    return value_summary(value)


def value_summary(value: Any) -> str:
    kind = type(value).__name__
    try:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, (int, float, str)):
            text = str(value)
            return text if len(text) <= 32 else text[:31] + "…"
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                dimensions = "×".join(str(int(size)) for size in tuple(shape))
            except Exception:
                dimensions = "shape unavailable"
            dtype = getattr(value, "dtype", None)
            suffix = f" · {dtype}" if dtype is not None else ""
            return f"{kind} · {dimensions or 'scalar'}{suffix}"
        if isinstance(value, Mapping):
            return f"{kind} · {len(value)} keys"
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return f"{kind} · {len(value)} items"
    except Exception:
        pass
    return kind


@dataclass(frozen=True)
class Node:
    """An ordinary function with named output ports."""

    name: str
    operation: Callable[..., Any]
    outputs: tuple[str, ...]
    signature: inspect.Signature
    cache: bool = field(default=True, compare=False)
    knobs: Mapping[str, Any] = field(default=MappingProxyType({}), compare=False)

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").strip().capitalize()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.operation(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return the callable contract without exposing the function object."""

        parameters = []
        for parameter in self.signature.parameters.values():
            parameters.append(
                {
                    "name": parameter.name,
                    "required": parameter.default is inspect.Parameter.empty,
                    "default": (
                        None
                        if parameter.default is inspect.Parameter.empty
                        else _public_value(parameter.default)
                    ),
                    "kind": parameter.kind.name.lower(),
                }
            )
        return {
            "name": self.name,
            "label": self.label,
            "operation": (
                f"{self.operation.__module__}.{self.operation.__qualname__}"
            ),
            "parameters": parameters,
            "outputs": self.outputs,
            "cache": self.cache,
            "knobs": {name: dict(spec) for name, spec in self.knobs.items()},
        }

    def __repr__(self) -> str:
        return f"Node({pprint.pformat(self.to_dict(), width=88, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


@dataclass(frozen=True)
class Run(Mapping[str, Any]):
    """One read-only top-level record; contained scientific values are not copied."""

    graph_name: str
    settings: Mapping[str, Any]
    outputs: Mapping[str, Any]
    order: tuple[str, ...]
    timings: Mapping[str, float]
    final_ports: tuple[str, ...]
    terminal_ports: tuple[str, ...]

    def __getitem__(self, port: str) -> Any:
        return self.outputs[port]

    def __iter__(self):
        return iter(self.outputs)

    def __len__(self) -> int:
        return len(self.outputs)

    @property
    def final(self) -> Any:
        if not self.final_ports:
            return None
        if len(self.final_ports) == 1:
            return self.outputs[self.final_ports[0]]
        return MappingProxyType(
            {port: self.outputs[port] for port in self.final_ports}
        )

    @property
    def seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def terminals(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {port: self.outputs[port] for port in self.terminal_ports}
        )

    def to_dict(self) -> dict[str, Any]:
        """Return bounded run metadata and shape-only value summaries.

        ``dict(run)`` remains the way to obtain the actual output values.
        This method is intentionally safe to print when outputs contain large
        scientific arrays.
        """

        return {
            "graph_name": self.graph_name,
            "settings": {
                name: _public_value(value)
                for name, value in self.settings.items()
            },
            "outputs": {
                name: value_summary(value)
                for name, value in self.outputs.items()
            },
            "order": self.order,
            "timings_seconds": dict(self.timings),
            "total_seconds": self.seconds,
            "final_ports": self.final_ports,
            "terminal_ports": self.terminal_ports,
            "actual_values": "dict(run)",
        }

    def __repr__(self) -> str:
        return f"Run({pprint.pformat(self.to_dict(), width=88, sort_dicts=False)})"

    def _repr_html_(self) -> str:
        return _dict_html(self.to_dict())


def _output_names(outputs: str | Sequence[str]) -> tuple[str, ...]:
    names = (outputs,) if isinstance(outputs, str) else tuple(outputs)
    if not names or any(not isinstance(name, str) or not name.isidentifier() for name in names):
        raise GraphError("node outputs must be one or more valid Python names")
    if len(names) != len(set(names)):
        raise GraphError("a node cannot declare the same output twice")
    return names


class _Knob:
    """Sentinel default carrying widget metadata; unwrapped to its real default at node build."""

    __slots__ = ("default", "spec")

    def __init__(self, default: Any, *, min: Any = None, max: Any = None, step: Any = None,
                 tier: str = "beginner", label: str | None = None, help: str | None = None,
                 choices: Sequence[Any] | None = None):
        self.default = default
        spec: dict[str, Any] = {"tier": tier}
        for key, value in (("min", min), ("max", max), ("step", step),
                           ("label", label), ("help", help)):
            if value is not None:
                spec[key] = value
        if choices is not None:
            spec["choices"] = list(choices)
        self.spec = spec


def knob(default: Any, *, min: Any = None, max: Any = None, step: Any = None,
         tier: str = "beginner", label: str | None = None, help: str | None = None,
         choices: Sequence[Any] | None = None) -> Any:
    """Annotate a node setting with widget metadata for ``graph.play`` and ``describe``.

    Use as the default of a setting parameter; ``run`` still sees the real ``default``.
    ``tier`` is ``"beginner"`` (shown by default) or ``"advanced"`` (revealed by a toggle).

        @graph.node(outputs="stats")
        def summarize(alpha=graph.knob(0.05, min=1e-3, max=0.2, tier="advanced",
                                       help="p-value threshold")):
            ...
    """
    return _Knob(default, min=min, max=max, step=step, tier=tier, label=label,
                 help=help, choices=choices)


def node(
    function: Callable[..., Any] | None = None,
    *,
    outputs: str | Sequence[str] | None = None,
    name: str | None = None,
    cache: bool = True,
) -> Node | Callable[[Callable[..., Any]], Node]:
    """Turn a function into a node.

    Parameters in the function signature become named input ports or settings.
    ``outputs`` must name the value or values returned by the function. Set
    ``cache=False`` for nodes whose output should never be reused (e.g. plots).
    """

    def decorate(operation: Callable[..., Any]) -> Node:
        if not callable(operation):
            raise TypeError("node expects a callable")
        if inspect.iscoroutinefunction(operation):
            raise GraphError("async node functions are not supported in notebook flows")
        node_name = name or operation.__name__
        if not node_name.isidentifier():
            raise GraphError(f"invalid node name: {node_name!r}")
        declared_outputs = operation.__name__ if outputs is None else outputs
        signature = inspect.signature(operation)
        knob_specs: dict[str, Any] = {}
        rebuilt = []
        for parameter in signature.parameters.values():
            if isinstance(parameter.default, _Knob):
                knob_specs[parameter.name] = dict(parameter.default.spec)
                parameter = parameter.replace(default=parameter.default.default)
            rebuilt.append(parameter)
        signature = signature.replace(parameters=rebuilt)
        unsupported = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in signature.parameters.values():
            if parameter.kind in unsupported:
                raise GraphError(
                    f"node {node_name!r} must use named parameters only; "
                    f"{parameter.name!r} is not supported"
                )
            if isinstance(parameter.default, (dict, list, set, bytearray)):
                raise GraphError(
                    f"node {node_name!r} setting {parameter.name!r} has a mutable "
                    "default; use None and create a fresh value inside the node"
                )
        return Node(
            name=node_name,
            operation=operation,
            outputs=_output_names(declared_outputs),
            signature=signature,
            cache=cache,
            knobs=MappingProxyType(knob_specs),
        )

    return decorate(function) if function is not None else decorate


_DRIVE_STORE_FACTORY: Callable[[str], Any] | None = None


def use_drive_cache(factory: Callable[[str], Any]) -> None:
    """Register a ``factory(graph_name) -> store`` so ``cache="drive"`` persists results.

    Keeps the graph package free of any data/framework dependency: the caller (e.g.
    the caller supplies the per-teammate Drive-backed store. Until this is set,
    ``cache="drive"`` degrades gracefully to a per-session in-memory cache.
    """
    global _DRIVE_STORE_FACTORY
    _DRIVE_STORE_FACTORY = factory


class _MemoryStore:
    """A per-session, in-process store used by cache="memory"."""

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}

    def get(self, spec: Any) -> Any:
        return self._entries.get(_spec_key(spec))

    def put(self, spec: Any, value: Any) -> None:
        self._entries[_spec_key(spec)] = value


def _spec_key(spec: Any) -> str:
    return repr(tuple(sorted(spec.items())))


def _make_store(cache: Any, graph_name: str) -> Any:
    if not cache:
        return None
    if cache == "memory":
        return _MemoryStore()
    if cache is True or cache == "drive":
        if _DRIVE_STORE_FACTORY is not None:
            return _DRIVE_STORE_FACTORY(graph_name)
        return _MemoryStore()
    if hasattr(cache, "get") and hasattr(cache, "put"):
        return cache
    raise GraphError(f"unknown cache option: {cache!r}; use 'drive', 'memory', a store, or None")


def _fingerprint_value(value: Any) -> str | None:
    """A stable token for a JSON-like setting value, or None if it is opaque."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        tokens = [_fingerprint_value(item) for item in value]
        if any(token is None for token in tokens):
            return None
        return "[" + ",".join(token for token in tokens if token is not None) + "]"
    if isinstance(value, Mapping):
        pairs = []
        for key in sorted(value, key=repr):
            token_key = _fingerprint_value(key)
            token_value = _fingerprint_value(value[key])
            if token_key is None or token_value is None:
                return None
            pairs.append(token_key + ":" + token_value)
        return "{" + ",".join(pairs) + "}"
    return None


def _cacheable_result(produced: Mapping[str, Any]) -> bool:
    return not any(_is_matplotlib_figure(value) for value in produced.values())
