from __future__ import annotations

import base64
import html
import inspect
import io
import pprint
from collections.abc import Mapping
from typing import Any


def _safe_error_text(error: BaseException) -> str:
    try:
        return str(error)
    except Exception:
        return f"{type(error).__name__} (message unavailable)"


_RESULT_TEXT_LIMIT = 12_000
_RESULT_MARKUP_LIMIT = 1_000_000
_RESULT_PNG_LIMIT = 8 * 1024 * 1024


def _oversized_result_html(kind: str, size: int, limit: int) -> str:
    return (
        "<pre style='margin:0;white-space:pre-wrap;overflow-wrap:anywhere'>"
        f"{html.escape(kind)} result omitted: {size:,} bytes exceeds the "
        f"{limit:,}-byte notebook display limit.</pre>"
    )


def _bounded_markup(payload: str, kind: str) -> str:
    size = len(payload.encode("utf-8"))
    if size > _RESULT_MARKUP_LIMIT:
        return _oversized_result_html(kind, size, _RESULT_MARKUP_LIMIT)
    return payload


def _rich_result_payload(value: Any, method_name: str) -> Any:
    try:
        inspect.getattr_static(value, method_name)
    except AttributeError:
        return None
    method = getattr(value, method_name)
    if not callable(method):
        return None
    payload = method()
    if isinstance(payload, tuple) and len(payload) == 2:
        payload = payload[0]
    return payload


def _is_matplotlib_figure(value: Any) -> bool:
    return any(
        current.__name__ == "Figure"
        and current.__module__.startswith("matplotlib.")
        for current in type(value).__mro__
    ) and callable(getattr(value, "savefig", None))


def _png_result_html(payload: bytes | bytearray | memoryview | str) -> str:
    if isinstance(payload, str):
        decoded = base64.b64decode(payload, validate=True)
    else:
        decoded = bytes(payload)
    if len(decoded) > _RESULT_PNG_LIMIT:
        return _oversized_result_html("PNG", len(decoded), _RESULT_PNG_LIMIT)
    encoded = base64.b64encode(decoded).decode("ascii")
    return (
        "<img alt='Rendered graph result' "
        "style='display:block;max-width:100%;height:auto' "
        f"src='data:image/png;base64,{encoded}'>"
    )


def _result_html(value: Any) -> str:
    """Render one result into HTML that can travel through a widget trait.

    ``widgets.Output`` relies on callback output capture, which is not reliable
    in every Colab frontend. This renderer instead serializes the visible
    result into the ``value`` trait of a plain HTML widget. Rich IPython
    representations are retained, while ordinary values are escaped and
    bounded before they enter the page.
    """

    body: str | None = None
    if _is_matplotlib_figure(value):
        image = io.BytesIO()
        dpi = min(float(getattr(value, "dpi", 100.0)), 150.0)
        value.savefig(image, format="png", bbox_inches="tight", dpi=dpi)
        body = _png_result_html(image.getvalue())
    else:
        rich_html = _rich_result_payload(value, "_repr_html_")
        if isinstance(rich_html, str):
            body = _bounded_markup(rich_html, "HTML")

        if body is None:
            svg = _rich_result_payload(value, "_repr_svg_")
            if isinstance(svg, bytes):
                svg = svg.decode("utf-8", errors="replace")
            if isinstance(svg, str):
                body = _bounded_markup(svg, "SVG")

        if body is None:
            png = _rich_result_payload(value, "_repr_png_")
            if isinstance(png, (bytes, bytearray, memoryview, str)):
                body = _png_result_html(png)

    if body is None:
        try:
            text = repr(value)
            if not isinstance(text, str):
                raise TypeError("repr returned a non-string value")
        except Exception:
            text = f"<{type(value).__name__} (representation unavailable)>"
        if len(text) > _RESULT_TEXT_LIMIT:
            text = text[: _RESULT_TEXT_LIMIT - 1] + "…"
        body = (
            "<pre style='margin:0;white-space:pre-wrap;overflow-wrap:anywhere'>"
            f"{html.escape(text)}</pre>"
        )

    return (
        "<div class='graph-result-view' role='region' aria-label='Graph result' "
        "style='box-sizing:border-box;max-width:100%;max-height:32rem;"
        "overflow:auto;padding:.65rem .75rem;border:1px solid rgba(127,127,127,.28);"
        "border-radius:6px'>"
        f"{body}</div>"
    )


def _dict_html(value: Mapping[str, Any]) -> str:
    rendered = pprint.pformat(dict(value), width=100, sort_dicts=False)
    return (
        "<pre style='margin:.3rem 0;padding:.7rem .8rem;max-width:1100px;"
        "max-height:34rem;overflow:auto;border:1px solid #7775;border-radius:7px;"
        "white-space:pre-wrap;overflow-wrap:anywhere'>"
        f"{html.escape(rendered)}</pre>"
    )
