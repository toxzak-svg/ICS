"""Shared CLI parsing for bounded vs full chain selection."""

from __future__ import annotations

import argparse


def parse_max_chains(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value < 1:
            raise argparse.ArgumentTypeError("max chains must be >= 1 or 'all'")
        return value
    normalized = value.strip().lower()
    if normalized in {"all", "none", "unlimited"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"max chains must be a positive integer or 'all', got {value!r}"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("max chains must be >= 1 or 'all'")
    return parsed


def format_max_chains(value: int | None) -> str:
    return "all" if value is None else str(value)
