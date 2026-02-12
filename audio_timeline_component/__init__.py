"""Streamlit component wrapper for the interactive audio timeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_NAME = "audio_timeline"


def _declare_component():
    dev_url = os.getenv("AUDIO_TIMELINE_DEV_URL")
    if dev_url:
        return components.declare_component(_COMPONENT_NAME, url=dev_url)

    build_dir = Path(__file__).resolve().parent / "frontend" / "build"
    return components.declare_component(_COMPONENT_NAME, path=str(build_dir))


_component_func = _declare_component()


def audio_timeline(
    *,
    audio_base64: str,
    audio_mime: str,
    duration_s: float,
    peaks: list[float],
    lanes: list[dict[str, Any]],
    class_colors: dict[str, Any],
    show_ground_truth: bool,
    show_predictions: bool,
    selected_classes: list[str],
    initial_viewport: dict[str, float] | None = None,
    key: str | None = None,
) -> Any:
    """Render the interactive audio timeline component."""
    return _component_func(
        audio_base64=audio_base64,
        audio_mime=audio_mime,
        duration_s=duration_s,
        peaks=peaks,
        lanes=lanes,
        class_colors=class_colors,
        show_ground_truth=show_ground_truth,
        show_predictions=show_predictions,
        selected_classes=selected_classes,
        initial_viewport=initial_viewport,
        key=key,
        default=None,
    )
