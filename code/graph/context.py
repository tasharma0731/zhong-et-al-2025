from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Protocol

from .types import Node, Run


class GraphContext(Protocol):
    name: str
    nodes: tuple[Node, ...]
    _connections: dict[tuple[str, str], tuple[str, str]]
    _settings: dict[str, tuple[str, inspect.Parameter]]
    _producer: dict[str, str]
    _nodes_by_name: dict[str, Node]

    @property
    def setting_defaults(self) -> Mapping[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...

    def _execution_nodes(self, until: str | None) -> tuple[Node, ...]: ...

    def run(
        self,
        *,
        until: str | None = None,
        cache: Any = None,
        **settings: Any,
    ) -> Run: ...

    @staticmethod
    def _widgets() -> Any: ...

    @staticmethod
    def _format_setting_value(value: Any) -> str: ...

    def _diagram_geometry(self) -> dict[str, Any]: ...

    def _diagram_html(self, *args: Any, **kwargs: Any) -> str: ...

    @staticmethod
    def _value_summary(value: Any) -> str: ...
