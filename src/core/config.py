"""Configuration loading and numeric coercion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml


_BOOL_TRUE = {"true", "yes", "on", "1"}
_BOOL_FALSE = {"false", "no", "off", "0"}


def _coerce_scalar(value: Any) -> Any:
    """Return a scalar with YAML-like numeric strings converted to native types."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    lower = text.lower()
    if lower in _BOOL_TRUE:
        return True
    if lower in _BOOL_FALSE:
        return False

    try:
        if any(ch in lower for ch in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return value


def coerce_config_types(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively coerce YAML-loaded config values into stable runtime types."""
    if config is None:
        return {}

    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: convert(val) for key, val in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return _coerce_scalar(value)

    return {key: convert(value) for key, value in config.items()}


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file and normalize scalar values for consumers."""
    with open(path, "r", encoding="utf-8") as fh:
        return coerce_config_types(yaml.safe_load(fh))

